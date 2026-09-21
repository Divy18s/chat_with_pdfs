"""Groq via OpenAI-compatible REST (urllib only — no SDK version hell)."""
import json, urllib.request, urllib.error
from django.conf import settings

def chat(messages, temperature=0.2, max_tokens=800):
    key = settings.GROQ_API_KEY
    if not key:
        raise RuntimeError('GROQ_API_KEY missing. Put your gsk_ key in chat_with_pdfs/.env')
    url = settings.GROQ_BASE_URL.rstrip('/') + '/chat/completions'
    payload = json.dumps({
        'model': settings.GROQ_MODEL,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0',
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode())
        return body['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:800]
        raise RuntimeError(f'Groq {e.code}: {detail}')

def stream_chat(messages, temperature=0.2, max_tokens=800):
    """Real-time token streaming generator via Groq native SSE API."""
    key = settings.GROQ_API_KEY
    if not key:
        raise RuntimeError('GROQ_API_KEY missing. Put your gsk_ key in chat_with_pdfs/.env')
    url = settings.GROQ_BASE_URL.rstrip('/') + '/chat/completions'
    payload = json.dumps({
        'model': settings.GROQ_MODEL,
        'messages': messages,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'stream': True,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={
        'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0',
    })
    try:
        r = urllib.request.urlopen(req, timeout=60)
        for line in r:
            l = line.decode('utf-8', errors='ignore').strip()
            if not l or not l.startswith('data:'):
                continue
            data_str = l[5:].strip()
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                token = chunk['choices'][0].get('delta', {}).get('content', '')
                if token:
                    yield token
            except Exception:
                continue
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:800]
        yield f"[Groq Error {e.code}: {detail}]"

SYSTEM = ('You are a PDF assistant. Answer naturally from the provided CONTEXT excerpts, '
          'like ChatGPT — plain sentences, no citation markers, no brackets, no footers. '
          'If the answer is not in the context, just say you cannot find it in the uploaded PDFs.')
