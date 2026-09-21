import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/
ROOT_DIR = BASE_DIR.parent  # chat_with_pdfs/ (where .env lives)

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-insecure-change-me')
DEBUG = os.environ.get('DJANGO_DEBUG', '1') == '1'
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', '*').split(',')

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'rag',
]

MIDDLEWARE = [
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = 'config.wsgi.application'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'rag' / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': []},
}]

STATIC_URL = '/static/'
DATA_DIR = BASE_DIR / 'data'
PDF_DIR = DATA_DIR / 'pdfs'
PDF_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# RAG / Groq config — .env may contain ONLY the raw gsk_ key (your case).
# We support both: `GROQ_API_KEY=gsk_...` and a file with just `gsk_...`.
def _load_groq_key():
    for p in [ROOT_DIR / '.env', BASE_DIR / '.env', Path('.env')]:
        try:
            if p.exists():
                txt = p.read_text().strip()
                if not txt:
                    continue
                for line in txt.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('GROQ_API_KEY'):
                        v = line.split('=', 1)[1].strip().strip('"').strip("'")
                        if v:
                            return v
                # fallback: bare key only
                tok = txt.split()[-1].split('=', 1)[-1].strip().strip('"').strip("'")
                if tok.startswith('gsk_'):
                    return tok
        except Exception:
            continue
    return os.environ.get('GROQ_API_KEY', '')

GROQ_API_KEY = os.environ.get('GROQ_API_KEY') or _load_groq_key()
# Default LLM model: Qwen 3.8 27B on Groq
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'qwen/qwen3.8-27b')
GROQ_BASE_URL = os.environ.get('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
MONGO_DB = os.environ.get('MONGO_DB', 'chatpdfs')
TOP_K = int(os.environ.get('RAG_TOP_K', '5'))
CHUNK_SIZE = int(os.environ.get('RAG_CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.environ.get('RAG_CHUNK_OVERLAP', '120'))
# Hybrid retrieval: dense MiniLM (384-d, local) + sparse TF-IDF fused with RRF.
# Swap EMBED_MODEL=BAAI/bge-m3 for quality (2GB download). Rerank off by default.
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
USE_DENSE = os.environ.get('USE_DENSE', '1')
USE_RERANK = os.environ.get('USE_RERANK', '0')
# Vision: CLIP ViT-B/32 page renders (true image vectors). Extra ~350MB download.
USE_VISION = os.environ.get('USE_VISION', '1')
# Captions: BLIP pixels->words so the LLM can describe figures. Extra ~500MB.
USE_CAPTION = os.environ.get('USE_CAPTION', '1')
