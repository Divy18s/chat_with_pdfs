"""Groq via OpenAI-compatible REST (urllib only — no SDK version hell)."""
import json, urllib.request, urllib.error
from django.conf import settings

def chat(messages, temperature=0.2, max_tokens=800, retries=1):
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
        if e.code == 429 and retries > 0:
            import time
            time.sleep(3)
            # If vision exceeded tokens, fallback to text-only context
            text_only_msgs = []
            for m in messages:
                if isinstance(m.get('content'), list):
                    txt_parts = [p['text'] for p in m['content'] if p.get('type') == 'text']
                    text_only_msgs.append({'role': m['role'], 'content': '\n'.join(txt_parts)})
                else:
                    text_only_msgs.append(m)
            return chat(text_only_msgs, temperature, max_tokens, retries=0)
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
        if e.code == 429:
            import time
            time.sleep(2)
            text_only_msgs = []
            for m in messages:
                if isinstance(m.get('content'), list):
                    txt_parts = [p['text'] for p in m['content'] if p.get('type') == 'text']
                    text_only_msgs.append({'role': m['role'], 'content': '\n'.join(txt_parts)})
                else:
                    text_only_msgs.append(m)
            try:
                for tok in stream_chat(text_only_msgs, temperature, max_tokens):
                    yield tok
                return
            except Exception:
                yield "\n[Rate limit reached on free tier. Please wait a moment before sending another query.]"
        else:
            detail = e.read().decode()[:800]
            yield f"[Groq Error {e.code}: {detail}]"

SYSTEM = ('You are an expert multimodal PDF assistant. You can read both text excerpts '
          'and attached page images/diagrams from the uploaded PDF documents. '
          'When answering: '
          '1. If page images/diagrams are attached, inspect them carefully to extract exact labels, '
          'diagram boxes, flowchart steps, arrows, chart values, tables, and visual relationships. '
          '2. Answer naturally and directly like ChatGPT — plain sentences, no citation brackets, no footers. '
          '3. If the answer cannot be determined from either the text excerpts or the attached images, '
          'clearly state that the information is not present in the document.')
