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

## Getting Started: Two Ways to Run

### Option 1: With Docker (Production Microservices Stack — Recommended)

Run the full containerized stack (Django Ninja API, Celery worker, MongoDB, Redis, and Next.js frontend):

1. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Add your GROQ_API_KEY in .env
   ```

2. **Launch Stack:**
   ```bash
   docker compose up --build
   ```
   * **Web Application & UI:** [http://localhost:8000/](http://localhost:8000/) (or Next.js at `:3000`)
   * **OpenAPI Documentation:** [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
   * **MongoDB:** `localhost:27017`
   * **Redis:** `localhost:6379`

3. **Horizontal Worker Scaling (Optional):**
   ```bash
   docker compose up --scale worker=3
   ```

---

### Option 2: Without Docker (Local Development / Bare Python)

Tested and verified on Python 3.11–3.14 on Windows & Linux with zero external daemon requirements:

1. **Configure Environment:**
   Create `.env` in the root folder with:
   ```bash
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=qwen/qwen3.8-27b
   ```

2. **Install & Run:**
   ```bash
   cd backend
   pip install -r requirements.txt
   python manage.py check
   python manage.py runserver 8000
   ```

3. **Open:** [http://localhost:8000/](http://localhost:8000/)
   * Runs **Embedded Qdrant** (`backend/data/qdrant`) locally out of the box with HNSW indexing and metadata filtering.
   * Connects to **MongoDB** if available (`MONGO_URI`), or transparently falls back to local JSON persistence (`backend/data/db_*.json`).

---

## Benchmark & Retrieval Evaluation

Evaluate retrieval accuracy, hit rates, and latency across test queries:
```bash
cd backend
python evaluate_rag.py
```
Outputs automated empirical comparisons across:
* **Hit Rate @ 1, 3, 5** (Did the correct chunk appear in the top 1, 3, or 5 results?)
* **MRR (Mean Reciprocal Rank)** (How close to rank #1 was the true source?)
* **Average Retrieval Latency (ms)** (p50/p95 search speed across sparse, dense, and visual pipelines)

---

## API Endpoints (Interactive OpenAPI Docs at `/api/docs`)

* `GET  /api/health` — System health (Qdrant, Dense, Vision, Groq, Mongo status)
* `GET  /api/chats` & `POST /api/chats` — Session management & multi-tenant isolation
* `POST /api/chats/{chat_id}/upload` — Threaded PDF ingestion & content-aware chunking
* `GET  /api/chats/{chat_id}/stream?q=...` — Real-time SSE token stream + citation chips
* `GET  /api/documents/{doc_id}/pdf` — Streaming PDF bytes for the split-screen viewer (`#page=N`)
* `DELETE /api/chats/{chat_id}` — Cascade deletion of chats, vectors, and documents

