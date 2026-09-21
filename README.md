# Chat with PDFs — Modern RAG (Django + MongoDB + Groq)

Legacy 2023 demo: `app.py` (Streamlit + FAISS, `requirements.txt`).
New stack in `backend/` + `frontend/` + Docker.

## Run locally (no Docker, no Node — tested on Python 3.14)
```
cd backend
pip install -r requirements.txt
python manage.py check
python manage.py runserver 8000
```
Open http://localhost:8000/ — upload PDF, ask, citations `[doc p.page]`.
- `.env`: root `.env` may contain ONLY the bare `gsk_...` key (supported). Or use `.env.example` format.
- Groq model default `openai/gpt-oss-120b` (verified on your key Sep 2026).
  Qwen (`qwen-qwq-32b`, `qwen3-32b`) + `llama-3.3-70b` are decommissioned / 404 on Groq free tier.
  Override: `set GROQ_MODEL=openai/gpt-oss-20b`.
- MongoDB optional: without it, JSON fallback in `backend/data/db_*.json` is used
  (`/api/health` shows `"mongo": false`). With Docker, real Mongo is used.

## API
- `GET /api/health`, `GET /api/documents`
- `POST /api/documents/upload` (multipart `file`)
- `POST /api/chat` `{"question": "...", "doc_id": "..."}` → `{answer, citations}`
- `GET /api/chat/stream?q=...&doc_id=...` → SSE word tokens + citations

## Docker
```
docker compose up --build
# frontend :3000, api :8000 (/api/docs), mongo :27017, redis :6379
docker compose up --scale worker=3   # demo horizontal scaling of ingestion
```
Services: `mongo` (source of truth) + `redis` (Celery broker) + `api` (Django+Ninja, sync ingest;
Celery worker target in compose for prod) + `frontend` (Next.js 15 scaffold) + `worker`.

## Retrieval: hybrid RRF (dense + sparse + vision) + per-type handling
- Dense: `all-MiniLM-L6-v2` 384-d via sentence-transformers (local, CPU), stored
  `backend/data/vectors/{doc_id}.npy` row-aligned to chunk idx. Override
  `EMBED_MODEL=BAAI/bge-m3` for quality. Deleted with the chat.
- Sparse: TF-IDF 1-2gram cosine. Fusion: RRF k=60 over rank lists.
- Vision: CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) image vectors of rendered
  pages (`{doc_id}.clip.npy`) + BLIP captions (`Salesforce/blip-image-captioning-base`)
  written into image-chunk text so the LLM can describe figures. Visual-intent
  queries (color/figure/chart/show) pin confident image hits first; text queries
  are unaffected. Flags: `USE_VISION`, `USE_CAPTION`.
- Rerank: optional CrossEncoder `ms-marco-MiniLM-L-6-v2` (`USE_RERANK=1`, extra download).
- Per type: text/code → raw text vectors; tables → markdown vectors (whole-table
  chunks); images → CLIP visual vector + BLIP caption + page-context words.
- Any signal failing → graceful fallback down to sparse-only. `/api/health`
  reports `dense/embed_model/embed_dim/vision/rerank`.
