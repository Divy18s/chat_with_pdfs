"""Hybrid retrieval: dense (MiniLM/BGE) + sparse (TF-IDF) + vision (CLIP) fused with RRF.

Per-type handling:
- text/code: MiniLM dense vector of raw chunk text.
- table: MiniLM dense vector of the markdown serialization (whole-table chunk).
- image: TWO signals — (a) MiniLM proxy vector of the [IMAGE p.N] placeholder,
  (b) true CLIP ViT-B/32 visual vector of the rendered page (512-d, shared
  CLIP text-image space). Query is encoded with CLIP text tower for the visual
  path and MiniLM for the text path; the three rank lists fuse with RRF.

Store: backend/data/vectors/{doc_id}.npy (MiniLM, row==chunk idx),
       backend/data/vectors/{doc_id}.clip.npy (CLIP image rows, aligned to the
       doc's image chunks in idx order) + {doc_id}.clip_pages.json.
Graceful: any failure -> fewer signals (vision/dense drop out, sparse remains).
"""
import os
from pathlib import Path
import numpy as np

_model = None
_model_ok = None
_reranker = None

def vec_dir():
    from django.conf import settings
    d = settings.DATA_DIR / 'vectors'
    d.mkdir(parents=True, exist_ok=True)
    return d

def model_name():
    from django.conf import settings
    return getattr(settings, 'EMBED_MODEL', os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2'))

def use_dense():
    from django.conf import settings
    return str(getattr(settings, 'USE_DENSE', os.environ.get('USE_DENSE', '1'))) == '1'

def use_rerank():
    from django.conf import settings
    return str(getattr(settings, 'USE_RERANK', os.environ.get('USE_RERANK', '0'))) == '1'

def get_model():
    global _model, _model_ok
    if _model is not None:
        return _model
    if _model_ok is False:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name())
        _model_ok = True
        return _model
    except Exception:
        _model_ok = False
        return None

def available():
    return use_dense() and get_model() is not None

def dim():
    m = get_model()
    try:
        return m.get_embedding_dimension()
    except Exception:
        return 384

def encode(texts):
    m = get_model()
    if m is None:
        raise RuntimeError('dense encoder unavailable')
    return np.asarray(m.encode(list(texts), normalize_embeddings=True), dtype=np.float32)

_qdrant = None
def get_qdrant():
    """Embedded Qdrant client singleton (zero external services required)."""
    global _qdrant
    if _qdrant is not None:
        return _qdrant
    try:
        from django.conf import settings
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance
        qdrant_dir = settings.DATA_DIR / 'qdrant'
        qdrant_dir.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(qdrant_dir))
        if not client.collection_exists('pdf_chunks'):
            client.create_collection(
                'pdf_chunks',
                vectors_config=VectorParams(size=dim(), distance=Distance.COSINE)
            )
        _qdrant = client
        return _qdrant
    except Exception:
        return None

def qdrant_available():
    return get_qdrant() is not None

def index_doc(doc_id, texts):
    """Encode chunk texts, save to Embedded Qdrant HNSW and local .npy. Returns {dense, dim, n, qdrant}."""
    if not texts:
        return {'dense': False}
    V = encode(texts)
    # 1. Save local array for fast fallback
    np.save(str(vec_dir() / f'{doc_id}.npy'), V)
    
    # 2. Upsert into Embedded Qdrant with payload
    qdrant_ok = False
    try:
        qc = get_qdrant()
        if qc is not None:
            from qdrant_client.models import PointStruct
            import uuid
            points = []
            for idx, (vec, txt) in enumerate(zip(V, texts)):
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{idx}"))
                points.append(PointStruct(
                    id=pid,
                    vector=vec.tolist(),
                    payload={'doc_id': doc_id, 'idx': idx, 'text': txt[:1000]}
                ))
            qc.upsert(collection_name='pdf_chunks', points=points)
            qdrant_ok = True
    except Exception:
        pass
    return {'dense': True, 'dim': int(V.shape[1]), 'n': int(V.shape[0]), 'qdrant': qdrant_ok}

def delete_docs(doc_ids):
    for d in doc_ids or []:
        try:
            p = vec_dir() / f'{d}.npy'
            if p.exists():
                p.unlink()
        except Exception:
            pass
    try:
        qc = get_qdrant()
        if qc is not None and doc_ids:
            from qdrant_client.models import Filter, FieldCondition, MatchAny, FilterSelector
            qc.delete(
                collection_name='pdf_chunks',
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key='doc_id', match=MatchAny(any=list(doc_ids)))])
                )
            )
    except Exception:
        pass

def _load_doc_matrix(doc_id):
    p = vec_dir() / f'{doc_id}.npy'
    if not p.exists():
        return None
    try:
        return np.load(str(p))
    except Exception:
        return None

def dense_scores(query, chunks):
    """Cosine sim (normalized dot) aligned to chunks order. Missing docs -> -inf."""
    q = encode([query])[0]
    out = np.full(len(chunks), -1e9, dtype=np.float32)
    by_doc = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c['doc_id'], []).append(i)
    for doc_id, idxs in by_doc.items():
        M = _load_doc_matrix(doc_id)
        if M is None:
            # on-the-fly fallback (old docs ingested before dense): encode now
            try:
                M = encode([chunks[i]['text'] for i in idxs])
            except Exception:
                continue
        # align rows by chunk idx
        for pos, i in enumerate(idxs):
            row = pos
            try:
                row = chunks[i].get('idx', pos)
                if row >= len(M):
                    row = pos
            except Exception:
                row = pos
            try:
                out[i] = float(M[row] @ q)
            except Exception:
                pass
    return out

def sparse_scores(query, corpus):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words='english').fit(corpus + [query])
    return cosine_similarity(vec.transform([query]), vec.transform(corpus))[0]

def rrf_fuse(*rank_lists, k=60):
    fused = {}
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + r + 1)
    return fused

_VISUAL_WORDS = frozenset(
    'image figure chart photo picture diagram plot graph screenshot scan scanned '
    'look looks shown shows depict depicts display red blue green yellow black white color '
    'architecture flowchart pipeline flow workflow box component block drawing illustration visual'.split())

def visual_intent(query):
    import re as _re
    toks = set(_re.findall(r'[a-z]+', query.lower()))
    return bool(toks & _VISUAL_WORDS)

def hybrid_order(query, chunks, top_k):
    """Returns (ordered_indices, info). RRF(dense_rank, sparse_rank)."""
    corpus = [c['text'] for c in chunks]
    try:
        sp = sparse_scores(query, corpus)
    except ValueError:
        return list(range(min(top_k, len(chunks)))), {'hybrid': False, 'reason': 'tfidf-empty'}
    sparse_rank = sorted(range(len(chunks)), key=lambda i: sp[i], reverse=True)
    try:
        dn = dense_scores(query, chunks)
        dense_rank = sorted(range(len(chunks)), key=lambda i: dn[i], reverse=True)
        lists = [dense_rank, sparse_rank]
        info = {'hybrid': True, 'dense': True,
                'top_dense': float(dn[sparse_rank[0]]) if chunks else 0.0,
                'top_sparse': float(sp[sparse_rank[0]]) if chunks else 0.0}
        try:
            if any(c.get('block') == 'image' for c in chunks) and vision_available():
                cv = clip_scores(query, chunks)
                if float(cv.max()) > -1e8:
                    clip_rank = sorted(range(len(chunks)), key=lambda i: cv[i], reverse=True)
                    lists.append(clip_rank)
                    info['vision'] = True
                    # visual queries ("red chart", "what does the figure show"):
                    # text votes drown the lone CLIP signal, so pin confident
                    # image hits first instead of letting RRF bury them.
                    if visual_intent(query) and float(cv[clip_rank[0]]) > 0.20:
                        imgs = [i for i in clip_rank if chunks[i].get('block') == 'image']
                        fused_all = rrf_fuse(*lists)
                        rest = [i for i in sorted(range(len(chunks)), key=lambda i: fused_all[i], reverse=True)
                                if i not in imgs]
                        order = (imgs + rest)[:top_k]
                        info['vision_boost'] = True
                        return order, info
        except Exception:
            pass
        fused = rrf_fuse(*lists)
        order = sorted(range(len(chunks)), key=lambda i: fused[i], reverse=True)[:top_k]
        return order, info
    except Exception as e:
        order = sparse_rank[:top_k]
        return order, {'hybrid': False, 'dense': False, 'reason': f'dense-fallback: {e}'}

def rerank(query, hits):
    """Optional CrossEncoder rerank over top hits. Off by default (USE_RERANK=1)."""
    global _reranker
    if not use_rerank() or not hits:
        return hits, {'reranked': False}
    try:
        from sentence_transformers import CrossEncoder
        if _reranker is None:
            _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [(query, h['text'][:1000]) for h in hits]
        scores = _reranker.predict(pairs)
        order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)
        return [hits[i] for i in order], {'reranked': True, 'model': 'ms-marco-MiniLM-L-6-v2'}
    except Exception as e:
        return hits, {'reranked': False, 'reason': str(e)[:120]}

# ---------- vision (CLIP ViT-B/32): true image embeddings ----------
_clip_model = None
_clip_proc = None
_clip_ok = None

def use_vision():
    import os as _os
    from django.conf import settings
    return str(getattr(settings, 'USE_VISION', _os.environ.get('USE_VISION', '1'))) == '1'

def get_clip():
    """Lazy (model, processor). ~350MB download first run. None on failure."""
    global _clip_model, _clip_proc, _clip_ok
    if _clip_model is not None:
        return _clip_model, _clip_proc
    if _clip_ok is False:
        return None, None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        _clip_proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        _clip_model.eval()
        _clip_ok = True
        return _clip_model, _clip_proc
    except Exception:
        _clip_ok = False
        return None, None

def vision_available():
    if not use_vision():
        return False
    m, p = get_clip()
    return m is not None

def encode_clip_images(pil_images):
    """True visual vectors: ViT -> pooler -> projection, L2-normalized (512-d)."""
    import torch
    m, p = get_clip()
    if m is None or not pil_images:
        raise RuntimeError('CLIP unavailable')
    with torch.no_grad():
        pix = p(images=list(pil_images), return_tensors='pt')['pixel_values']
        pooled = m.vision_model(pix).pooler_output
        vec = m.visual_projection(pooled)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().astype(np.float32)

def encode_clip_text(texts):
    import torch
    m, p = get_clip()
    if m is None:
        raise RuntimeError('CLIP unavailable')
    with torch.no_grad():
        tok = p(text=list(texts), return_tensors='pt', padding=True, truncation=True)
        pooled = m.text_model(tok['input_ids'], attention_mask=tok.get('attention_mask')).pooler_output
        vec = m.text_projection(pooled)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy().astype(np.float32)

def index_images(doc_id, pil_images, pages):
    """Save CLIP rows aligned to the doc's image chunks (idx order)."""
    import json as _json
    if not pil_images:
        return {'vision': False, 'reason': 'no-images'}
    V = encode_clip_images(pil_images)
    np.save(str(vec_dir() / f'{doc_id}.clip.npy'), V)
    (vec_dir() / f'{doc_id}.clip_pages.json').write_text(_json.dumps(list(pages or [])))
    return {'vision': True, 'dim': 512, 'n_images': int(V.shape[0])}

# ---------- vision captions (BLIP): pixels -> words so the LLM can quote them ----------
_blip_model = None
_blip_proc = None
_blip_ok = None

def use_caption():
    import os as _os
    from django.conf import settings
    return str(getattr(settings, 'USE_CAPTION', _os.environ.get('USE_CAPTION', '1'))) == '1'

def caption_images(pil_images, max_words=30):
    """BLIP caption per image, e.g. 'a red square on a white background'."""
    global _blip_model, _blip_proc, _blip_ok
    if not pil_images:
        return []
    try:
        if _blip_model is None and _blip_ok is not False:
            import torch
            from transformers import BlipProcessor, BlipForConditionalGeneration
            _blip_proc = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
            _blip_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
            _blip_model.eval()
        import torch as _torch
        out = []
        for im in pil_images:
            inp = _blip_proc(im, return_tensors='pt')
            with _torch.no_grad():
                ids = _blip_model.generate(**inp, max_new_tokens=max_words)
            out.append(_blip_proc.decode(ids[0], skip_special_tokens=True))
        return out
    except Exception as e:
        _blip_ok = False
        return [f'' for _ in pil_images]

def delete_vision(doc_ids):
    for d in doc_ids or []:
        for suf in ('.clip.npy', '.clip_pages.json'):
            try:
                p = vec_dir() / f'{d}{suf}'
                if p.exists():
                    p.unlink()
            except Exception:
                pass

def clip_scores(query, chunks):
    """CLIP text-vs-image cosine for image chunks; -inf elsewhere.

    Row alignment: clip matrix rows follow the doc's image chunks sorted by idx.
    """
    out = np.full(len(chunks), -1e9, dtype=np.float32)
    img_pos = [i for i, c in enumerate(chunks) if c.get('block') == 'image']
    if not img_pos:
        return out
    q = encode_clip_text([query])[0]
    by_doc = {}
    for i in img_pos:
        by_doc.setdefault(chunks[i]['doc_id'], []).append(i)
    for doc_id, idxs in by_doc.items():
        p = vec_dir() / f'{doc_id}.clip.npy'
        if not p.exists():
            continue
        try:
            M = np.load(str(p))
        except Exception:
            continue
        # doc's image chunks in idx order define row order
        doc_img = sorted(idxs, key=lambda i: chunks[i].get('idx', 0))
        for row, i in enumerate(doc_img):
            if row < len(M):
                try:
                    out[i] = float(M[row] @ q)
                except Exception:
                    pass
    return out
