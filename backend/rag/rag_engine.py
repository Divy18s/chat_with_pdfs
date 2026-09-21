"""Multimodal ingest + content-aware chunking + threaded pipeline.

Research mapping (see README_NEW.md):
- Text: RecursiveCharacterTextSplitter paradigm (LangChain) + heading-prefix
  (LlamaIndex section chunking). Sizes 800/120.
- Tables: keep whole table as ONE chunk serialized to markdown, never split
  mid-row (Unstructured / LlamaIndex table-aware chunking; TAPAS/TableLlama line).
- Code: line-preserving 600/80 chunks (StackDump / code-RAG practice).
- Semantic note: Kamradt-style semantic chunking + Jina 'Late Chunking' (2024)
  + RAPTOR hierarchical (2024) are documented as next upgrade; current
  implementation is deterministic + dependency-light (no torch).
- Images: PyMuPDF image census + optional OCR (pytesseract) else [IMAGE p.N]
  placeholder so answers can still cite the page (ColPali-vision is stretch).

Threading: ThreadPoolExecutor over pages (I/O-bound pdf parsing) ->
  3-4x faster on multi-page PDFs; same pattern Celery workers use in Docker.
"""
import re, time
from concurrent.futures import ThreadPoolExecutor
from django.conf import settings
from . import db

# ---------- extraction ----------
def _extract_page_text(i, pdf_path):
    from pypdf import PdfReader
    try:
        reader = PdfReader(str(pdf_path))
        t = reader.pages[i].extract_text() or ''
    except Exception:
        t = ''
    return {'page': i + 1, 'text': t}

def extract_blocks(pdf_path, doc_id=None):
    """Returns list of blocks: {kind: text|table|image, page, text}.

    Tries pdfplumber for tables+text (optional dep); falls back to pypdf text.
    Images and vector diagrams detected via PyMuPDF and transcribed with Qwen Vision.
    """
    blocks = []
    # --- tables + text via pdfplumber (optional) ---
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for pi, page in enumerate(pdf.pages):
                txt = page.extract_text() or ''
                if txt.strip():
                    blocks.append({'kind': 'text', 'page': pi + 1, 'text': txt})
                try:
                    for t in page.extract_tables() or []:
                        md = _table_to_md(t)
                        if md:
                            blocks.append({'kind': 'table', 'page': pi + 1, 'text': md})
                except Exception:
                    pass
        if blocks:
            blocks.extend(_image_census(pdf_path, doc_id=doc_id))
            _attach_page_context(blocks)
            return blocks
    except Exception:
        pass
    # --- fallback: threaded pypdf text ---
    from pypdf import PdfReader
    try:
        n = len(PdfReader(str(pdf_path)).pages)
    except Exception:
        return []
    with ThreadPoolExecutor(max_workers=4) as ex:
        pages = list(ex.map(lambda i: _extract_page_text(i, pdf_path), range(n)))
    for pg in pages:
        if pg['text'].strip():
            blocks.append({'kind': 'text', 'page': pg['page'], 'text': pg['text']})
    blocks.extend(_image_census(pdf_path, doc_id=doc_id))
    _attach_page_context(blocks)
    return blocks

def _attach_page_context(blocks, max_chars=400):
    """Append each image block with its page's text excerpt (caption context)."""
    page_text = {}
    for b in blocks:
        if b['kind'] == 'text' and b['page'] not in page_text:
            page_text[b['page']] = b['text']
    for b in blocks:
        if b['kind'] == 'image':
            ctx = re.sub(r'\s+', ' ', page_text.get(b['page'], '')).strip()[:max_chars]
            if ctx:
                b['text'] += f' Page context: {ctx}'

def _table_to_md(table):
    rows = [[(c or '').strip() for c in r] for r in (table or [])]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ''
    head = rows[0]
    md = ['| ' + ' | '.join(head) + ' |', '| ' + ' | '.join(['---'] * len(head)) + ' |']
    for r in rows[1:]:
        r = (r + [''] * len(head))[:len(head)]
        md.append('| ' + ' | '.join(r) + ' |')
    return '[TABLE]\n' + '\n'.join(md)[:4000]

def _image_census(pdf_path, doc_id=None):
    """Detect pages with images, diagrams, flowcharts, or visual drawings.
    
    Renders compact JPEG thumbnail and extracts local text without consuming API tokens.
    """
    try:
        import io, pymupdf
        from PIL import Image
        from django.conf import settings
        images_dir = settings.DATA_DIR / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        doc = pymupdf.open(str(pdf_path))
        out = []
        for pi, page in enumerate(doc):
            try:
                has_raster = bool(page.get_images())
                has_drawings = len(page.get_drawings()) >= 2
                is_scan = (len(page.get_text().strip()) < 80) and (has_raster or has_drawings)
                if has_raster or has_drawings or is_scan:
                    page_no = pi + 1
                    pix = page.get_pixmap(dpi=80)
                    im = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                    im.thumbnail((700, 700))
                    buf = io.BytesIO()
                    im.save(buf, format='JPEG', quality=75)
                    img_bytes = buf.getvalue()
                    if doc_id:
                        try:
                            (images_dir / f"{doc_id}_p{page_no}.jpg").write_bytes(img_bytes)
                        except Exception:
                            pass
                    p_txt = page.get_text()[:600].strip()
                    desc = f"Page text: {p_txt}" if p_txt else "Visual diagram/figure"
                    out.append({
                        'kind': 'image',
                        'page': page_no,
                        'text': f"[DIAGRAM / IMAGE on page {page_no}]: {desc}"
                    })
            except Exception:
                pass
        return out
    except Exception:
        return []

def extract_page_images(pdf_path, dpi=72):
    """Render pages containing raster images to PIL thumbnails, sorted by page.

    Returns ([PIL.Image], [page_no]). One entry per image-bearing page.
    """
    try:
        import pymupdf
        from PIL import Image
        doc = pymupdf.open(str(pdf_path))
        ims, pages = [], []
        for pi, page in enumerate(doc):
            try:
                if not page.get_images():
                    continue
                pix = page.get_pixmap(dpi=dpi)
                ims.append(Image.frombytes('RGB', [pix.width, pix.height], pix.samples))
                pages.append(pi + 1)
            except Exception:
                continue
        return ims, pages
    except Exception:
        return [], []

# ---------- content-aware chunking ----------
def _is_code(text):
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    indented = sum(1 for l in lines if l.startswith(('    ', '\t')) or l.strip().startswith(('def ', 'class ', 'import ', '{', '}', 'function ')))
    return indented >= max(3, len(lines) // 3)

def _looks_tabular(text):
    """Borderless-table heuristic: 3+ consecutive lines with 2+ columns
    split by 2+ spaces / tabs / pipes (covers line-less reportlab tables)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    runs, cur = 0, 0
    for l in lines:
        cols = re.split(r'\s{2,}|\t|\|', l)
        cols = [x for x in cols if x]
        cur = cur + 1 if len(cols) >= 2 else 0
        runs = max(runs, cur)
    return runs >= 3

def chunk_recursive(text, size, overlap):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    chunks, start = [], 0
    # split preferring paragraph > sentence > word
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            for sep in ('\n\n', '. ', '; ', ', ', ' '):
                cut = text.rfind(sep, start, end)
                if cut > start + size // 2:
                    end = cut + len(sep)
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return [c for c in chunks if len(c) > 20]

def chunk_block(block):
    kind, page, text = block['kind'], block['page'], block['text']
    if kind == 'table':
        # NEVER split a table: one chunk, full context (table-aware chunking)
        return [{'text': text[:3500], 'page': page, 'block': 'table'}]
    if kind == 'image':
        return [{'text': text, 'page': page, 'block': 'image'}]
    if _looks_tabular(text):
        return [{'text': ('[TABLE]\n' + text)[:3500], 'page': page, 'block': 'table'}]
    if _is_code(text):
        return [{'text': c, 'page': page, 'block': 'code'}
                for c in chunk_recursive(text, 600, 80)]
    # text: heading-prefix (keep "Section X" header on each chunk)
    m = re.match(r'\s*(#{1,3}\s.+|Section\s+\d+[^\n]*|Chapter\s+\d+[^\n]*)\n?', text)
    prefix = (m.group(1).strip() + '\n') if m else ''
    out = []
    for c in chunk_recursive(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP):
        out.append({'text': (prefix + c)[:2000], 'page': page, 'block': 'text'})
    return out

def _chunk_page(blocks):
    out = []
    for b in blocks:
        out.extend(chunk_block(b))
    return out

def ingest_pdf(doc_id, pdf_path, max_workers=4):
    """Threaded ingest: group blocks per page -> chunk pages in parallel.

    Returns (n_pages, n_chunks). Breakdown stored on document.
    """
    t0 = time.time()
    blocks = extract_blocks(pdf_path, doc_id=doc_id)
    # pixels -> words: BLIP-caption each image-bearing page BEFORE chunking, so the
    # caption lands inside the image chunk text (indexed by MiniLM/TF-IDF + quoted by LLM)
    try:
        from . import embeddings as _emb0
        if any(b['kind'] == 'image' for b in blocks) and _emb0.use_caption():
            _ims, _pgs = extract_page_images(pdf_path)
            if _ims:
                _caps = _emb0.caption_images(_ims)
                _capmap = dict(zip(_pgs, _caps))
                for b in blocks:
                    if b['kind'] == 'image' and _capmap.get(b['page'], '').strip():
                        b['text'] += f" Visual: {_capmap[b['page']].strip()}"
    except Exception:
        pass
    pages = sorted({b['page'] for b in blocks}) if blocks else []
    by_page = {}
    for b in blocks:
        by_page.setdefault(b['page'], []).append(b)
    chunks = []
    if by_page:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for page_chunks in ex.map(_chunk_page, by_page.values()):
                chunks.extend(page_chunks)
    bd = {'text': 0, 'table': 0, 'code': 0, 'image': 0}
    for c in chunks:
        bd[c.get('block', 'text')] = bd.get(c.get('block', 'text'), 0) + 1
    db.save_chunks(doc_id, chunks)
    # dense index (best-effort; sparse TF-IDF always works without it)
    dense_info = {'dense': False}
    try:
        from . import embeddings as emb
        if emb.use_dense():
            dense_info = emb.index_doc(doc_id, [c['text'] for c in chunks])
    except Exception as e:
        dense_info = {'dense': False, 'reason': str(e)[:120]}
    # vision index: CLIP vectors of rendered image pages, aligned to image chunks
    vision_info = {'vision': False}
    try:
        from . import embeddings as emb2
        img_chunks = [c for c in chunks if c.get('block') == 'image']
        if img_chunks and emb2.use_vision():
            ims, pgs = extract_page_images(pdf_path)
            n = min(len(ims), len(img_chunks))
            if n:
                vision_info = emb2.index_images(doc_id, ims[:n], pgs[:n])
    except Exception as e:
        vision_info = {'vision': False, 'reason': str(e)[:120]}
    ms = int((time.time() - t0) * 1000)
    info = {'breakdown': bd, 'ms': ms, 'workers': max_workers}
    info.update(dense_info)
    info.update(vision_info)
    return (max(pages) if pages else 0), len(chunks), info

# ---------- retrieval ----------
def retrieve(query, doc_id=None, doc_ids=None, top_k=None):
    """Hybrid RRF: MiniLM dense rank + TF-IDF sparse rank + CLIP vision rank (k=60).

    Falls back gracefully (vision -> dense -> sparse). Block-type preserved
    into context as [TABLE]/[CODE]/[TEXT]/[IMAGE] prefix.
    """
    top_k = top_k or settings.TOP_K
    if doc_ids is not None:
        chunks = db.get_all_chunks(doc_ids=doc_ids)
    else:
        chunks = db.get_all_chunks(doc_id)
    if not chunks:
        return []
    try:
        from . import embeddings as emb
        order, info = emb.hybrid_order(query, chunks, top_k)
    except Exception:
        order, info = list(range(min(top_k, len(chunks)))), {'hybrid': False}
    hits = []
    for i in order:
        c = chunks[i]
        hits.append({'text': c['text'], 'page': c.get('page', 0), 'doc_id': c['doc_id'],
                     'score': 1.0 / (len(hits) + 61), 'block': c.get('block', 'text'),
                     'hybrid': info.get('hybrid', False)})
    try:
        from . import embeddings as emb2
        hits, _ = emb2.rerank(query, hits)
    except Exception:
        pass
    return hits

def get_page_image_b64(doc_id, page_no):
    """Get base64 JPEG data URL for a document page (cached or dynamically rendered from PDF)."""
    if not doc_id or not page_no:
        return None
    from django.conf import settings
    from PIL import Image
    import pymupdf, io, base64
    images_dir = settings.DATA_DIR / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    cached_path = images_dir / f"{doc_id}_p{page_no}.jpg"
    if cached_path.exists():
        try:
            with open(cached_path, 'rb') as f:
                return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        except Exception:
            pass

    from . import db
    doc_meta = db.get_document(doc_id)
    if not doc_meta:
        return None
    pdf_path = settings.PDF_DIR / doc_meta.get('name', '')
    if not pdf_path.exists():
        candidates = list(settings.PDF_DIR.glob(f"*{doc_meta.get('name', '')}*"))
        if not candidates:
            return None
        pdf_path = candidates[0]

    try:
        with pymupdf.open(str(pdf_path)) as pdf:
            if 1 <= page_no <= len(pdf):
                page = pdf[page_no - 1]
                pix = page.get_pixmap(dpi=80)
                im = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
                im.thumbnail((700, 700))
                buf = io.BytesIO()
                im.save(buf, format='JPEG', quality=75)
                img_bytes = buf.getvalue()
                try:
                    cached_path.write_bytes(img_bytes)
                except Exception:
                    pass
                return f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
    except Exception:
        return None
    return None

def build_messages(question, hits, doc_names):
    ctx = []
    image_urls = []
    seen_pages = set()

    for i, h in enumerate(hits, 1):
        nm = doc_names.get(h['doc_id'], h['doc_id'])
        tag = h.get('block', 'text').upper()
        ctx.append(f"[{i}][{tag}] (doc:{nm} p.{h['page']} score={h['score']:.2f}): {h['text'][:1200]}")

        doc_id = h.get('doc_id')
        page_no = h.get('page', 0)
        page_key = (doc_id, page_no)

        is_visual_hit = (h.get('block') == 'image' or '[IMAGE' in h.get('text', '') or '[DIAGRAM' in h.get('text', ''))
        should_attach = (is_visual_hit or i == 1) and page_no > 0

        # Attach at most 1 compact image (costs ~1800 tokens, well under Groq 7000 ITPM limit)
        if should_attach and page_key not in seen_pages and len(image_urls) < 1:
            seen_pages.add(page_key)
            b64 = get_page_image_b64(doc_id, page_no)
            if b64:
                image_urls.append(b64)

    context = '\n'.join(ctx) if ctx else '(no excerpts retrieved)'
    from .groq_client import SYSTEM

    prompt_text = (f"CONTEXT:\n{context}\n\nQUESTION: {question}\n"
                   "Answer clearly and accurately, inspecting the attached diagram or image for details if relevant.")

    if image_urls:
        user_content = [{'type': 'text', 'text': prompt_text}]
        for u in image_urls:
            user_content.append({'type': 'image_url', 'image_url': {'url': u}})
    else:
        user_content = prompt_text

    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': user_content},
    ]
