"""MongoDB with transparent local-JSON fallback (so tests pass without Docker).

Collections (mongo) / files (fallback):
- chats: {id, title, doc_ids[], created_at}
- documents: {id, name, pages, chunks, chat_id, breakdown{}, created_at}
- chunks: {doc_id, idx, text, page, block}
- messages: {session_id/chat_id, role, content, citations[], at}
"""
import json, uuid, time
from datetime import datetime, timezone
from pathlib import Path
from django.conf import settings

_docs_json = settings.DATA_DIR / 'db_documents.json'
_chunks_json = settings.DATA_DIR / 'db_chunks.json'
_msgs_json = settings.DATA_DIR / 'db_messages.json'
_chats_json = settings.DATA_DIR / 'db_chats.json'

def _load(p):
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return []

def _save(p, rows):
    p.write_text(json.dumps(rows, indent=1))

def _now():
    return datetime.now(timezone.utc).isoformat()

_mongo = None
def mongo_db():
    global _mongo
    if _mongo is not None:
        return _mongo
    try:
        from pymongo import MongoClient
        c = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=1200)
        c.server_info()
        _mongo = c[settings.MONGO_DB]
        return _mongo
    except Exception:
        _mongo = False
        return False

def using_mongo():
    return bool(mongo_db())

# ---------- chats (sessions) ----------
def create_chat(title=''):
    cid = uuid.uuid4().hex[:12]
    chat = {'id': cid, 'title': title or f'Chat {cid[:6]}', 'doc_ids': [], 'created_at': _now()}
    db = mongo_db()
    if db:
        db.chats.insert_one(dict(chat))
    else:
        rows = _load(_chats_json); rows.append(chat); _save(_chats_json, rows)
    return chat

def list_chats():
    db = mongo_db()
    if db:
        return list(db.chats.find({}, {'_id': 0}).sort('created_at', -1))
    return list(reversed(_load(_chats_json)))

def get_chat(chat_id):
    db = mongo_db()
    if db:
        return db.chats.find_one({'id': chat_id}, {'_id': 0})
    for c in _load(_chats_json):
        if c['id'] == chat_id:
            return c
    return None

def add_doc_to_chat(chat_id, doc_id, title_hint=''):
    db = mongo_db()
    if db:
        db.chats.update_one({'id': chat_id}, {'$addToSet': {'doc_ids': doc_id}})
        if title_hint:
            ch = db.chats.find_one({'id': chat_id}, {'title': 1})
            if ch and ch.get('title', '').startswith('Chat '):
                db.chats.update_one({'id': chat_id}, {'$set': {'title': title_hint[:40]}})
    else:
        rows = _load(_chats_json)
        for c in rows:
            if c['id'] == chat_id:
                if doc_id not in c.get('doc_ids', []):
                    c.setdefault('doc_ids', []).append(doc_id)
                if title_hint and c.get('title', '').startswith('Chat '):
                    c['title'] = title_hint[:40]
        _save(_chats_json, rows)

# ---------- documents (scoped to chat) ----------
def save_document(name, n_pages, n_chunks, chat_id='', breakdown=None):
    doc = {'id': uuid.uuid4().hex[:12], 'name': name, 'pages': n_pages,
           'chunks': n_chunks, 'chat_id': chat_id or '',
           'breakdown': breakdown or {}, 'created_at': _now()}
    db = mongo_db()
    if db:
        db.documents.insert_one(dict(doc))
    else:
        rows = _load(_docs_json); rows.append(doc); _save(_docs_json, rows)
    return doc

def update_doc_counts(doc_id, pages, chunks, breakdown=None):
    db = mongo_db()
    if db:
        u = {'pages': pages, 'chunks': chunks}
        if breakdown is not None:
            u['breakdown'] = breakdown
        db.documents.update_one({'id': doc_id}, {'$set': u})
    else:
        rows = _load(_docs_json)
        for r in rows:
            if r['id'] == doc_id:
                r['pages'], r['chunks'] = pages, chunks
                if breakdown is not None:
                    r['breakdown'] = breakdown
        _save(_docs_json, rows)

def list_documents(chat_id=None):
    db = mongo_db()
    if db:
        q = {} if not chat_id else {'chat_id': chat_id}
        return list(db.documents.find(q, {'_id': 0}).sort('created_at', -1))
    rows = _load(_docs_json)
    if chat_id:
        rows = [r for r in rows if r.get('chat_id') == chat_id]
    return list(reversed(rows))

# ---------- chunks ----------
def save_chunks(doc_id, chunks):
    """chunks: list of {text, page, block}"""
    t0 = time.time()
    rows = [{'doc_id': doc_id, 'idx': i, 'text': c['text'],
             'page': c.get('page', 0), 'block': c.get('block', 'text')}
            for i, c in enumerate(chunks)]
    db = mongo_db()
    if db:
        if rows:
            db.chunks.insert_many(rows)
    else:
        old = [r for r in _load(_chunks_json) if r['doc_id'] != doc_id]
        old.extend(rows)
        _save(_chunks_json, old)
    return {'saved': len(rows), 'ms': int((time.time() - t0) * 1000)}

def get_all_chunks(doc_id=None, doc_ids=None):
    db = mongo_db()
    if db:
        if doc_ids:
            return list(db.chunks.find({'doc_id': {'$in': doc_ids}}, {'_id': 0}))
        q = {} if not doc_id else {'doc_id': doc_id}
        return list(db.chunks.find(q, {'_id': 0}))
    rows = _load(_chunks_json)
    if doc_ids:
        return [r for r in rows if r['doc_id'] in doc_ids]
    return rows if not doc_id else [r for r in rows if r['doc_id'] == doc_id]

def get_chunks_for_chat(chat_id):
    ch = get_chat(chat_id)
    if not ch:
        return []
    return get_all_chunks(doc_ids=ch.get('doc_ids', []))

# ---------- messages (per chat) ----------
def save_message(role, content, doc_id='', citations=None, chat_id=''):
    msg = {'role': role, 'content': content, 'doc_id': doc_id,
           'chat_id': chat_id or '', 'citations': citations or [], 'at': _now()}
    db = mongo_db()
    if db:
        db.messages.insert_one(dict(msg))
    else:
        rows = _load(_msgs_json); rows.append(msg); _save(_msgs_json, rows[-500:])

def get_messages(chat_id, limit=100):
    db = mongo_db()
    if db:
        return list(db.messages.find({'chat_id': chat_id}, {'_id': 0}).sort('at', 1).limit(limit))
    return [m for m in _load(_msgs_json) if m.get('chat_id') == chat_id][-limit:]

# ---------- delete chat + its PDFs/chunks/messages ----------
def delete_chat(chat_id):
    """Delete chat, its documents, chunks, messages and PDF files from disk."""
    docs = list_documents(chat_id)
    doc_ids = [d['id'] for d in docs]
    files_removed = []
    for d in docs:
        try:
            p = settings.PDF_DIR / d.get('name', '')
            if p.exists() and p.is_file():
                p.unlink()
                files_removed.append(d.get('name', ''))
        except Exception:
            pass
    try:
        from . import embeddings as emb
        emb.delete_docs(doc_ids)
        emb.delete_vision(doc_ids)
    except Exception:
        pass
    db = mongo_db()
    if db:
        if doc_ids:
            db.chunks.delete_many({'doc_id': {'$in': doc_ids}})
        db.documents.delete_many({'chat_id': chat_id})
        db.messages.delete_many({'chat_id': chat_id})
        db.chats.delete_one({'id': chat_id})
    else:
        _save(_chunks_json, [r for r in _load(_chunks_json) if r.get('doc_id') not in doc_ids])
        _save(_docs_json, [r for r in _load(_docs_json) if r.get('chat_id') != chat_id])
        _save(_msgs_json, [m for m in _load(_msgs_json) if m.get('chat_id') != chat_id])
        _save(_chats_json, [c for c in _load(_chats_json) if c.get('id') != chat_id])
    return {'chat_id': chat_id, 'docs_deleted': len(doc_ids), 'files_removed': files_removed}
