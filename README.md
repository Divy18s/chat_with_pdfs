# Chat with PDFs — Enterprise RAG & SDE Platform

Modernized high-performance RAG system featuring Django Ninja, Groq real-time streaming, Embedded Qdrant HNSW vector search, and an interactive NotebookLM-style PDF previewer with citation navigation.

Legacy 2023 prototype: `app.py` (Streamlit + FAISS).
Modern 2026 stack: `backend/` + `frontend/` + Docker.

---

## System Architecture & Highlights

* **System Design & Architecture:** Built a containerized microservices app using **Docker Compose**, **Django Ninja** for stateless APIs, and **Celery/Redis** for background processing; implemented **polyglot persistence** (MongoDB for chats + Qdrant HNSW for vectors) with isolated multi-tenant workspaces and complete **cascade deletion**.
* **Smart Hybrid Search & Reliability:** Created a 3-way hybrid search combining **MiniLM** (semantic meaning), **TF-IDF** (exact keywords), and **CLIP** (diagrams) using **Reciprocal Rank Fusion ($k=60$)**, engineered with a multi-tier **graceful degradation ladder** (automatic fallbacks) so the app never crashes.
* **Vision, Reasoning & Frontend UX:** Paired **BLIP diagram captions** with **Qwen 27B reasoning** under **token-budget downscaling ($700\times700$)** to prevent API rate limits; built a responsive **Tailwind CSS split-screen UI** with real-time SSE streaming, clickable PDF page citations, and an automated **evaluation benchmark** (Hit Rate/MRR).

---

## Quick Start (No Docker, No Node Required)

Tested and verified on Python 3.11–3.14 on Windows & Linux.

```bash
cd backend
pip install -r requirements.txt
python manage.py check
python manage.py runserver 8000
```
Open **http://localhost:8000/** in your browser:
* **Upload PDFs**: Threaded parsing, table extraction, and embedded HNSW indexing.
* **Real-Time Streaming**: Low-latency token-by-token generation (TTFT < 250ms).
* **Click-to-Page Citations**: Click any `[📄 Source p.X]` chip in the chat to jump the split-screen PDF viewer directly to that page.

---

## Configuration (`.env`)

Put your Groq API key in `.env` (or copy from `.env.example`):
```bash
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

### Database & Vector Storage (Zero-Docker Ready)
* **Vector Database**: Runs **Embedded Qdrant** (`DATA_DIR/qdrant`) locally out of the box with HNSW indexing and metadata filtering.
* **Document & Chat Database**: Connects to **MongoDB Atlas** (cloud) or local MongoDB if available (`MONGO_URI`). If unreachable, transparently falls back to local JSON persistence (`DATA_DIR/db_*.json`).

---

## Benchmark & Retrieval Evaluation

Evaluate retrieval accuracy, hit rates, and latency:
```bash
cd backend
python evaluate_rag.py
```
Outputs automated comparison across:
* **Hit Rate @ 1, 3, 5**
* **MRR (Mean Reciprocal Rank)**
* **Average Retrieval Latency (ms)**

---

## API Endpoints (OpenAPI Docs at `/api/docs`)

* `GET  /api/health` — System health (Qdrant, Dense, Vision, Groq, Mongo status)
* `GET  /api/chats` & `POST /api/chats` — Session management
* `POST /api/chats/{chat_id}/upload` — Threaded PDF ingestion
* `GET  /api/chats/{chat_id}/stream?q=...` — Real-time SSE token stream + citations
* `GET  /api/documents/{doc_id}/pdf` — Inline PDF serving for interactive viewer

---

## Docker Deployment (Optional)

```bash
docker compose up --build
# frontend :3000, api :8000 (/api/docs), mongo :27017, redis :6379
docker compose up --scale worker=3   # horizontal worker scaling
```
