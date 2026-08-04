# 🛠️ Papeer — Project Build Log & Engineering Flow

> A behind-the-scenes look at how **Papeer** (a production-grade Research Paper RAG system) was designed, built, broken, debugged, and hardened — step by step.

This document exists for one reason: to show *how* I think, not just *what* I shipped. Anyone can list "Built a RAG app" on a resume. This log walks through the real decisions, the dead ends, the failures, and the fixes.

---

## 📍 Table of Contents

1. [Phase 0 — Project Setup](#phase-0--project-setup)
2. [Phase 1 — Backend Skeleton](#phase-1--backend-skeleton)
3. [Phase 2 — Documenting Intent First](#phase-2--documenting-intent-first)
4. [Phase 3 — Building the Core Backend](#phase-3--building-the-core-backend)
5. [Phase 4 — The Pluggable LLM Architecture](#phase-4--the-pluggable-llm-architecture)
6. [Phase 5 — The Retrieval Problem (and the Fix)](#phase-5--the-retrieval-problem-and-the-fix)
7. [Phase 6 — Evaluation with RAGAS](#phase-6--evaluation-with-ragas)
8. [Phase 7 — Hybrid Search (BM25 + MMR)](#phase-7--hybrid-search-bm25--mmr)
9. [What's Next](#-whats-next)

---

## Phase 0 — Project Setup

Before writing a single line of application code, the project was set up the way any production repo should be — reproducible, versioned, and clean from commit #1.

```bash
# 1. Initialize and clone the repo
git init papeer
git clone <repo-url>
cd papeer

# 2. Create an isolated virtual environment (macOS)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Keep the repo clean
touch .gitignore   # venv/, __pycache__/, .env, *.pyc, .DS_Store, local vector stores, etc.
```

**Why this matters:** starting with `venv` isolation and a `.gitignore` from day one avoids the classic "it works on my machine" trap and keeps secrets (API keys) out of version control — a small habit that signals engineering discipline early.

---

## Phase 1 — Backend Skeleton

Rather than writing code inside a single monolithic script, the backend was scaffolded first so every future module had an obvious home:

```
backend/
├── __init__.py
├── models.py          # Pydantic schemas / data models
├── paper_loader.py     # Document ingestion & chunking
├── vector_store.py     # Embeddings + Qdrant vector DB logic
├── rag_graph.py         # LangGraph orchestration (the "brain")
└── btw_handler.py       # LLM handler ("behind-the-wire" model logic)
```

**Why this matters:** designing the module boundaries *before* the implementation forces you to think about separation of concerns — ingestion, storage, orchestration, and inference are all independent, swappable pieces. This is what makes the system "pluggable" later.

---

## Phase 2 — Documenting Intent First

Before writing the retrieval or generation logic, an `about_project.md` was written to capture the *intended* design: goals, scope, and constraints. Writing the spec before the implementation kept the build focused and gave a reference point to check the system against once it was live.

---

## Phase 3 — Building the Core Backend

Backend development moved in a deliberate order — data in, data stored, then data reasoned over:

1. **`paper_loader.py`** — document loader responsible for ingesting research papers and preparing them for chunking/embedding.
2. **`vector_store.py`** — wired up to **Qdrant** for vector storage, alongside a `.env` file for secrets:
   ```env
   QDRANT_URL=...
   QDRANT_API_KEY=...
   HUGGINGFACEHUB_API_TOKEN=...
   ```
3. **`models.py`** and **`btw_handler.py`** — the LLM handling layer. For local development, **`qwen2.5:3b` via Ollama** was chosen deliberately: it's lightweight enough to run locally without hitting API rate limits, while still being capable enough to validate the pipeline end-to-end before spending API credits.

---

## Phase 4 — The Pluggable LLM Architecture

A key design decision: **the application logic should never care which model is answering the question.** Whether it's a local Ollama model during development or a managed cloud model in production, the graph, the retriever, and the UI stay untouched. Only the factory layer changes.

```
                       .env
                        │
                        ▼
        ┌───────────────────────────┐
        │                           │
        ▼                           ▼
 llm_factory.py           embedding_factory.py
        │                           │
        ▼                           ▼
   rag_graph.py               vector_store.py
        │                           │
        ▼                           ▼
 btw_handler.py           CacheBackedEmbeddings
        │                           │
        └─────────────┬─────────────┘
                       ▼
                   LangGraph
                       │
                       ▼
                   Streamlit
```

On top of this, **stateless conversational memory** was added to the chat history — each session tracks its own context without leaking state across users, keeping the graph horizontally scalable.

**Why this matters:** this is the difference between a notebook demo and a system designed for production. Swapping `qwen2.5:3b` for a hosted model later required zero changes to `rag_graph.py` — only the factory config.

---

## Phase 5 — The Retrieval Problem (and the Fix)

At inference time, the first real bug surfaced: the retriever wasn't hallucinating — which would've been the *easier* problem — it was doing something worse. It was correctly identifying when it didn't know, and responding **"I don't have the answer"** far too often, even when the answer clearly existed in the source paper.

**Root cause:** naive chunking meant that individual retrieved chunks lacked enough surrounding context for the LLM to confidently ground its answer, so the model played it safe and refused to answer.

**The fix — Parent Document Retrieval:**
- Small "child" chunks are embedded and indexed in Qdrant for precise semantic search.
- Their full "parent" documents are persisted separately (initially in a local file store), and once a child chunk matches, the **full parent context** is handed to the LLM instead of the narrow fragment.

The result: retrieval accuracy jumped immediately, and the model started answering correctly and confidently instead of defaulting to "I don't know." This was the single biggest quality unlock in the whole project.

---

## Phase 6 — Evaluation with RAGAS

Gut-feeling ("it seems to work now") isn't good enough for a production-grade system, so the pipeline was evaluated quantitatively using **RAGAS**, across 5 core metrics (faithfulness, answer relevancy, context precision, context recall, and context relevancy).

One practical constraint: **Ollama doesn't support parallel test execution**, which makes batch evaluation painfully slow. So evaluation was decoupled from generation — the local Ollama model stayed in place for actual RAG answers, while the **Gemini model was used purely as the RAGAS judge**, enabling parallelized, faster evaluation runs.

Initial results landed around **~80% Answer Relevancy** — a strong baseline that highlighted exactly where the pipeline still had room to improve (precision on ambiguous, multi-paper queries).

---

## Phase 7 — Hybrid Search (BM25 + MMR)

With a solid relevancy baseline established, retrieval was upgraded from pure dense vector search to a **hybrid approach**:

- **BM25** — classic sparse keyword search, which excels at exact term/entity matching (crucial for research papers full of specific terminology, author names, and technical jargon that dense embeddings can blur).
- **MMR (Maximal Marginal Relevance)** — re-ranks results to reduce redundancy and surface *diverse* supporting passages instead of five near-duplicate chunks, which matters a lot for research paper Q&A where an answer often needs synthesis across sections.

Combining sparse (BM25) and dense (embedding) retrieval gave the system the best of both: precision on exact terms, and semantic understanding on conceptual questions.

---

## 🔭 What's Next

A few natural next steps that weren't fully captured above but are worth calling out as the project matures:

- [ ] **Testing** — unit tests for `paper_loader.py` chunking logic and integration tests for the LangGraph pipeline.
- [ ] **`.env.example`** — a template committed to the repo so new contributors know exactly which secrets to set up.
- [ ] **Dockerfile** — containerize the backend + Streamlit app for one-command reproducibility.
- [ ] **CI pipeline** — lint + test on every push (GitHub Actions).
- [ ] **Cross-encoder reranking** — layering a reranker on top of the hybrid retriever for a further precision boost.
- [ ] **Persistent parent store migration** — moving the parent docstore from local file storage to a managed Postgres instance for durability across deployments.
- [ ] **Deployment** — Streamlit Cloud / Docker deployment with a public demo link.
- [ ] **Security hardening** — rate limiting, PII detection, and audit logging on the inference endpoints.

---

