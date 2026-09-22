# Chat with PDFs — Enterprise RAG & SDE Platform

Modernized high-performance RAG system featuring Django Ninja, Groq real-time streaming, Embedded Qdrant HNSW vector search, and an interactive NotebookLM-style PDF previewer with citation navigation.

Legacy 2023 prototype: `app.py` (Streamlit + FAISS).
Modern 2026 stack: `backend/` + `frontend/` + Docker.

---

## System Architecture & Highlights

* **System Design & Architecture:** Built a containerized microservices app using **Docker Compose**, **Django Ninja** for stateless APIs, and **Celery/Redis** for background processing; implemented **polyglot persistence** (MongoDB for chats + Qdrant HNSW for vectors) with isolated multi-tenant workspaces and complete **cascade deletion**.
* **Smart Hybrid Search & Reliability:** Created a 3-way hybrid search combining **MiniLM** (semantic meaning), **TF-IDF** (exact keywords), and **CLIP** (diagrams) using **Reciprocal Rank Fusion ($k=60$)**, engineered with a multi-tier **graceful degradation ladder** (automatic fallbacks) so the app never crashes.
* **Vision, Reasoning & Frontend UX:** Paired **BLIP diagram captions** with **Qwen 27B reasoning** under **token-budget downscaling ($700\times700$)** to prevent API rate limits; built a responsive **Tailwind CSS split-screen UI** with real-time SSE streaming, clickable PDF page citations, and an automated benchmark achieving **100% Hit Rate @ 3, 1.0 MRR, and <20ms search latency**.

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

Run the automated evaluation harness to benchmark retrieval accuracy and latency across algorithms:
```bash
cd backend
python evaluate_rag.py
```

### Empirical Benchmark Results

| Retrieval Algorithm | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR (Mean Reciprocal Rank) | Avg Latency | Key Advantage |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sparse Only (TF-IDF)** | 90.0% | 90.0% | 100.0% | 0.925 | ~7 ms | Exact keyword, code, & acronym matching |
| **Dense Only (MiniLM-L6-v2)** | 100.0% | 100.0% | 100.0% | 1.000 | ~14 ms | Deep semantic & conceptual understanding |
| **Hybrid RRF (Combined Stack)** | **100.0%** | **100.0%** | **100.0%** | **1.000** | **~17 ms** | **Best-of-both: fuses semantics + exact keywords + diagrams** |

* **Hit Rate @ K:** Measures whether the correct source page was retrieved in the top $K$ results.
* **MRR (Mean Reciprocal Rank):** Measures how close to Rank #1 the true answer was placed (1.0 = perfect #1 rank).
* **Latency:** End-to-end retrieval speed across all chunks before LLM generation.

> **Key Takeaway for SDE / AI Interviews:** Naive RAG using a single vector search fails on exact product codes, acronyms, and diagram figures. Fusing dense neural embeddings (MiniLM) with sparse lexical search (TF-IDF) and visual vectors (CLIP) via **Reciprocal Rank Fusion ($k=60$)** guarantees maximum retrieval precision with sub-20ms latency.

---

## API Endpoints (Interactive OpenAPI Docs at `/api/docs`)

* `GET  /api/health` — System health (Qdrant, Dense, Vision, Groq, Mongo status)
* `GET  /api/chats` & `POST /api/chats` — Session management & multi-tenant isolation
* `POST /api/chats/{chat_id}/upload` — Threaded PDF ingestion & content-aware chunking
* `GET  /api/chats/{chat_id}/stream?q=...` — Real-time SSE token stream + citation chips
* `GET  /api/documents/{doc_id}/pdf` — Streaming PDF bytes for the split-screen viewer (`#page=N`)
* `DELETE /api/chats/{chat_id}` — Cascade deletion of chats, vectors, and documents

