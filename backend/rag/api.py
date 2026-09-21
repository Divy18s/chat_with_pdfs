import json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from django.conf import settings
from django.http import StreamingHttpResponse
from ninja import NinjaAPI, File, Schema
from ninja.files import UploadedFile
from . import db
from .rag_engine import ingest_pdf, retrieve, build_messages
from . import groq_client

api = NinjaAPI(urls_namespace='rag')

class ChatIn(Schema):
    question: str = ''
    doc_id: str = ''

class NewChat(Schema):
    title: str = ''

def _store_upload(f, chat_id=''):
    dest = settings.PDF_DIR / f.name
    i = 1
    while dest.exists():
        dest = settings.PDF_DIR / f"{Path(f.name).stem}_{i}.pdf"
        i += 1
    with open(dest, 'wb') as out:
        for chunk in f.chunks():
            out.write(chunk)
    return dest

def _ingest_many(files, chat_id=''):
    """Parallel multi-file ingest (thread per file; pages threaded inside)."""
    t0 = time.time()
    dests = [_store_upload(f) for f in files]
    docs = [db.save_document(d.name, 0, 0, chat_id=chat_id) for d in dests]

    def _one(args):
        doc, dest = args
        try:
            n_pages, n_chunks, info = ingest_pdf(doc['id'], dest)
            db.update_doc_counts(doc['id'], n_pages, n_chunks, info['breakdown'])
            doc.update(pages=n_pages, chunks=n_chunks, breakdown=info['breakdown'],
                       dense=info.get('dense', False), dim=info.get('dim', 0),
                       vision=info.get('vision', False))
            if chat_id:
                db.add_doc_to_chat(chat_id, doc['id'], title_hint=doc['name'])
            return {'document': doc, 'ingest_ms': info['ms'], 'workers': info['workers'],
                    'dense': info.get('dense', False), 'vision': info.get('vision', False)}
        except Exception as e:
            return {'error': f'{doc["name"]}: PDF parse failed: {e}'}

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(docs)))) as ex:
        results = list(ex.map(_one, zip(docs, dests)))
    return {'results': results, 'total_ms': int((time.time() - t0) * 1000)}

@api.get('/health')
def health(request):
    try:
        from . import embeddings as emb
        dense, dim = emb.available(), (emb.dim() if emb.available() else 0)
        vision = emb.vision_available()
        qdrant = emb.qdrant_available()
    except Exception:
        dense, dim, vision, qdrant = False, 0, False, False
    return {'ok': True, 'model': settings.GROQ_MODEL, 'mongo': db.using_mongo(),
            'groq_key': bool(settings.GROQ_API_KEY),
            'dense': dense, 'embed_model': settings.EMBED_MODEL, 'embed_dim': dim,
            'vision': vision, 'rerank': settings.USE_RERANK == '1', 'qdrant': qdrant}

# ---------- chats ----------
@api.get('/chats')
def chats(request):
    return {'chats': db.list_chats()}

@api.post('/chats')
def new_chat(request, data: NewChat):
    return {'chat': db.create_chat(data.title)}

@api.get('/chats/{chat_id}')
def get_chat(request, chat_id: str):
    ch = db.get_chat(chat_id)
    if not ch:
        return api.create_response(request, {'error': 'chat not found'}, status=404)
    return {'chat': ch, 'documents': db.list_documents(chat_id),
            'messages': db.get_messages(chat_id)}

@api.get('/chats/{chat_id}/documents')
def chat_docs(request, chat_id: str):
    return {'documents': db.list_documents(chat_id)}

@api.get('/chats/{chat_id}/messages')
def chat_msgs(request, chat_id: str):
    return {'messages': db.get_messages(chat_id)}

@api.delete('/chats/{chat_id}')
def del_chat(request, chat_id: str):
    if not db.get_chat(chat_id):
        return api.create_response(request, {'error': 'chat not found'}, status=404)
    return db.delete_chat(chat_id)

@api.post('/chats/{chat_id}/upload')
def chat_upload(request, chat_id: str, file: UploadedFile = File(...)):
    if not db.get_chat(chat_id):
        return api.create_response(request, {'error': 'chat not found'}, status=404)
    if not file.name.lower().endswith('.pdf'):
        return api.create_response(request, {'error': 'Only PDFs'}, status=400)
    out = _ingest_many([file], chat_id)
    r = out['results'][0]
    if 'error' in r:
        return api.create_response(request, r, status=400)
    r['total_ms'] = out['total_ms']
    return r

@api.post('/chats/{chat_id}/chat')
def chat_in(request, chat_id: str, data: ChatIn):
    ch = db.get_chat(chat_id)
    if not ch:
        return api.create_response(request, {'error': 'chat not found'}, status=404)
    q = (data.question or '').strip()
    if not q:
        return {'error': 'empty question'}
    hits = retrieve(q, doc_ids=ch.get('doc_ids', []))
    names = {d['id']: d['name'] for d in db.list_documents(chat_id)}
    msgs = build_messages(q, hits, names)
    answer = groq_client.chat(msgs)
    cites = [{'doc': names.get(h['doc_id'], h['doc_id']), 'page': h['page'],
              'block': h.get('block', 'text'), 'score': round(h['score'], 3)} for h in hits]
    db.save_message('user', q, '', [], chat_id)
    db.save_message('assistant', answer, '', cites, chat_id)
    if len(db.get_messages(chat_id)) <= 2 and answer:
        pass
    return {'answer': answer, 'citations': cites, 'hits': len(hits)}

@api.get('/chats/{chat_id}/stream')
def chat_stream_s(request, chat_id: str, q: str = ''):
    ch = db.get_chat(chat_id) or {}
    hits = retrieve(q, doc_ids=ch.get('doc_ids', []))
    names = {d['id']: d['name'] for d in db.list_documents(chat_id)}
    cites = [{'doc': names.get(h['doc_id'], h['doc_id']), 'doc_id': h['doc_id'], 'page': h['page'],
              'block': h.get('block', 'text'), 'score': round(h['score'], 3)} for h in hits]

    def gen():
        yield f"data: {json.dumps({'citations': cites})}\n\n"
        full_tokens = []
        if q:
            db.save_message('user', q, '', [], chat_id)
            messages = build_messages(q, hits, names)
            try:
                for token in groq_client.stream_chat(messages):
                    full_tokens.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except Exception as e:
                err_msg = f"[Streaming error: {e}]"
                full_tokens.append(err_msg)
                yield f"data: {json.dumps({'token': err_msg})}\n\n"
            full_answer = ''.join(full_tokens)
            db.save_message('assistant', full_answer, '', cites, chat_id)
        yield 'data: [DONE]\n\n'
    return StreamingHttpResponse(gen(), content_type='text/event-stream')

# ---------- PDF serving for interactive viewer ----------
@api.get('/documents/{doc_id}/pdf')
def get_document_pdf(request, doc_id: str):
    doc = db.get_document(doc_id)
    if not doc:
        return api.create_response(request, {'error': 'Document not found'}, status=404)
    file_path = settings.PDF_DIR / doc['name']
    if not file_path.exists():
        candidates = list(settings.PDF_DIR.glob(f"*{doc['name']}*"))
        if not candidates:
            return api.create_response(request, {'error': 'PDF file missing on disk'}, status=404)
        file_path = candidates[0]
    from django.http import FileResponse
    res = FileResponse(open(file_path, 'rb'), content_type='application/pdf')
    res['Content-Disposition'] = f'inline; filename="{doc["name"]}"'
    return res

# ---------- legacy global endpoints (back-compat) ----------
@api.get('/documents')
def docs(request):
    return {'documents': db.list_documents()}

@api.post('/documents/upload')
def upload(request, file: UploadedFile = File(...)):
    if not file.name.lower().endswith('.pdf'):
        return api.create_response(request, {'error': 'Only PDFs'}, status=400)
    out = _ingest_many([file], '')
    r = out['results'][0]
    if 'error' in r:
        return api.create_response(request, r, status=400)
    return {'document': r['document']}

@api.post('/documents/upload-many')
def upload_many(request, files: list[UploadedFile] = File(...)):
    """Parallel multi-PDF upload: threads per file (multithreaded RAG ingest)."""
    return _ingest_many(files, '')

@api.post('/chat')
def chat(request, data: ChatIn):
    q = (data.question or '').strip()
    if not q:
        return {'error': 'empty question'}
    hits = retrieve(q, data.doc_id or None)
    names = {d['id']: d['name'] for d in db.list_documents()}
    answer = groq_client.chat(build_messages(q, hits, names))
    cites = [{'doc': names.get(h['doc_id'], h['doc_id']), 'doc_id': h['doc_id'], 'page': h['page'],
              'block': h.get('block', 'text'), 'score': round(h['score'], 3)} for h in hits]
    db.save_message('user', q, data.doc_id or '', [])
    db.save_message('assistant', answer, data.doc_id or '', cites)
    return {'answer': answer, 'citations': cites, 'hits': len(hits)}

@api.get('/chat/stream')
def chat_stream(request, q: str = '', doc_id: str = ''):
    hits = retrieve(q, doc_id or None)
    names = {d['id']: d['name'] for d in db.list_documents()}
    cites = [{'doc': names.get(h['doc_id'], h['doc_id']), 'doc_id': h['doc_id'], 'page': h['page']} for h in hits]

    def gen():
        yield f"data: {json.dumps({'citations': cites})}\n\n"
        full_tokens = []
        if q:
            db.save_message('user', q, doc_id or '', [])
            messages = build_messages(q, hits, names)
            try:
                for token in groq_client.stream_chat(messages):
                    full_tokens.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
            except Exception as e:
                err_msg = f"[Streaming error: {e}]"
                full_tokens.append(err_msg)
                yield f"data: {json.dumps({'token': err_msg})}\n\n"
            full_answer = ''.join(full_tokens)
            db.save_message('assistant', full_answer, doc_id or '', cites)
        yield 'data: [DONE]\n\n'
    return StreamingHttpResponse(gen(), content_type='text/event-stream')
