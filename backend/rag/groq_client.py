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

SYSTEM = ('You are a PDF assistant. Answer naturally from the provided CONTEXT excerpts, '
          'like ChatGPT — plain sentences, no citation markers, no brackets, no footers. '
          'If the answer is not in the context, just say you cannot find it in the uploaded PDFs.')
