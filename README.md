# 📄 Papeer — Research Paper RAG System

**Ask questions. Get grounded, cited answers straight from the research papers themselves — not the model's imagination.**

Papeer is a production-minded Retrieval-Augmented Generation (RAG) system purpose-built for interacting with academic research papers. It's designed around a pluggable architecture that runs entirely on free tooling, while being production-ready to swap providers with a single env-var change.

---

## 🚀 Why Papeer

Most RAG demos stop at "upload a PDF, ask a question." Papeer goes further — it's built to actually answer correctly on dense, jargon-heavy research papers, and it's evaluated, not just demoed:

- ❌ **The naive version failed silently.** Early on, the retriever wasn't hallucinating — it was under-confident, frequently answering *"I don't have the answer"* even when the paper clearly had it.
- ✅ **Fixed with Parent Document Retrieval**, giving the LLM full contextual passages instead of narrow, disconnected chunks.
- 📊 **Evaluated with RAGAS** across 4 core dimensions: **0.93 Faithfulness**, **0.95 Answer Relevancy**, **0.94 Context Precision**, and **0.91 Context Recall**.
- 🔀 **Hybrid retrieval** (BM25 + dense embeddings + MMR) so exact terminology *and* conceptual questions both get answered well.

Read the full engineering story in [`projectflow.md`](./projectflow.md).

---

## ✨ Key Features

| Feature | Description |
|---|---|
| ⚡ **Single-Provider LLM** | Groq (`openai/gpt-oss-20b`) handles every pipeline task — routing, retrieval, relevancy, rewrite, generation — fast, free, no daily cap |
| 🧠 **Semantic Cache** | BetterDB + Valkey vector similarity cache (threshold 0.5) — similar questions answered instantly from cache with hit/miss telemetry and cost-saved display |
| 📚 **Parent Document Retrieval** | Retrieves precise small chunks, answers with full parent context |
| 🔍 **Hybrid Search** | BM25 (sparse/keyword) + dense embeddings, re-ranked with MMR for diverse, non-redundant results |
| 🔒 **Private /btw Side-Channel** | Off-record questions rendered in a distinct ephemeral block, never saved to session history or LangGraph checkpointer |
| 🧵 **Per-Session Isolation** | Each conversation gets its own Qdrant collection, local docstore, and SQLite checkpoint — state survives server restarts |
| 📈 **RAGAS-Evaluated** | Quantitatively benchmarked retrieval & generation quality, not just vibes |
| ⚡ **LangGraph Orchestration** | The RAG pipeline is modeled as an explicit, inspectable graph rather than a black-box chain |
| 🖥️ **Streamlit UI** | Lightweight, fast interface with cache hit/miss badges and per-turn graph state inspection |

---

## 🏗️ Architecture

```
                          User Query
                               │
                        Fast-path check
                       (greeting/name?)
                               │
               ┌───────────────┴───────────────┐
               │ Yes                           │ No
               ▼                               ▼
        Direct answer                   Groq Router LLM
        (zero API call)              (intent classification)
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                 ▼
                       retrieve      verify_claim     direct_answer
                          │                │                 │
                    Groq agent        Tavily web          Groq LLM
                   (tool calls)      + arXiv search
                          │
                    Qdrant hybrid
                    retrieval
                    (BM25 + MMR)
                          │
                    Groq generates
                    final answer
```

```
                       .env
                        │
                        ▼
        ┌───────────────────────────┐
        │                           │
        ▼                           ▼
 llm_factory.py           embedding_factory.py
   (Groq / Ollama / vLLM)    (BGE / HuggingFace)
        │                           │
        ▼                           ▼
   rag_graph.py               vector_store.py
  (Intent Router)                (Qdrant)
        │                           │
        ▼                           ▼
 btw_handler.py           CacheBackedEmbeddings
        │                           │
        └─────────────┬─────────────┘
                       ▼
              semantic_cache.py
              (BetterDB + Valkey)
                       │
                   LangGraph
                       │
                   Streamlit
```

**Design principle:** A single Groq model (`openai/gpt-oss-20b`) handles all pipeline tasks — routing, retrieval agent, relevancy checking, query rewriting, claim verification, and answer generation. A fast-path pattern matcher intercepts conversational queries before any API call is made. A semantic cache (BetterDB + Valkey) serves similar queries instantly from cache, reducing latency and API calls further.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| **Orchestration** | LangChain, LangGraph |
| **LLM** | Groq `openai/gpt-oss-20b` — all pipeline tasks |
| **Semantic Cache** | BetterDB + Valkey (vector similarity, threshold 0.5) |
| **Evaluation Judge** | Groq `openai/gpt-oss-120b` (decoupled RAGAS evaluation) |
| **Embeddings** | HuggingFace `BAAI/bge-base-en-v1.5` (`CacheBackedEmbeddings`) |
| **Vector Store** | Qdrant |
| **Retrieval** | Parent Document Retrieval + Hybrid Search (BM25 + Dense + MMR) |
| **Evaluation** | RAGAS (4-metric suite: 0.93 Faithfulness, 0.95 Answer Relevancy) |
| **Frontend** | Streamlit |
| **Environment** | Python 3, `venv` |

---

## 📁 Project Structure

```
papeer/
├── Backend/
│   ├── __init__.py
│   ├── config.py            # Central config (embedding dim, etc.)
│   ├── models.py            # Pydantic schemas for all LLM structured outputs
│   ├── paper_loader.py      # Document ingestion (PDF, TXT, MD, URL, arXiv)
│   ├── embedding_factory.py # Pluggable embedding model factory
│   ├── llm_factory.py       # LLM factory (Groq / Ollama / vLLM / OpenAI)
│   ├── vector_store.py      # Qdrant vector store + hybrid retrieval
│   ├── rag_graph.py         # LangGraph agentic pipeline orchestration
│   ├── btw_handler.py       # /btw private side-channel handler
│   └── semantic_cache.py    # BetterDB + Valkey semantic cache layer
├── Documents/               # Sample research papers
├── about_project.md         # Original design spec
├── projectflow.md           # Full engineering build narrative
├── graph.png                # Current LangGraph pipeline visualisation
├── requirements.txt
├── .env                     # API keys and config (never committed)
├── .gitignore
└── README.md
```

---

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd papeer
```

### 2. Set up a virtual environment (macOS)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
> Using Windows/Linux? Activate with `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux).

### 3. Start Valkey (for semantic cache)
```bash
# macOS with Homebrew
brew install valkey && brew services start valkey

# Or via Docker
docker run -d -p 6379:6379 valkey/valkey
```

### 4. Configure environment variables
Create a `.env` file in the root directory:
```env
# ── Required ──────────────────────────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key              # https://console.groq.com (free tier)
GROQ_MODEL=openai/gpt-oss-20b
TAVILY_API_KEY=your_tavily_api_key          # https://tavily.com

QDRANT_URL=your_qdrant_cluster_url          # https://cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_PROVIDER=groq

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_PROVIDER=huggingface
HF_EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# ── Semantic cache ────────────────────────────────────────────────────────
VALKEY_HOST=localhost
VALKEY_PORT=6379
SEMANTIC_CACHE_THRESHOLD=0.5
```

### 5. Run the app
```bash
streamlit run app.py
```

---

## 📊 Evaluation

The retrieval and generation pipeline is quantitatively benchmarked using **RAGAS** across 4 core evaluation metrics:

| Metric | Score | Description |
|---|---|---|
| **Faithfulness** | **0.93** (93%) | Answers are grounded strictly in retrieved paper context, eliminating hallucinations |
| **Answer Relevancy** | **0.95** (95%) | Generated responses directly and completely answer user questions |
| **Context Precision** | **0.94** (94%) | Retrieved parent chunks prioritize signal over noise, ranking relevant facts highest |
| **Context Recall** | **0.91** (91%) | Pipeline retrieves all essential reference information required for the ground truth |

> **Decoupled Judge Architecture**: Evaluation is decoupled from pipeline generation. The pipeline uses `openai/gpt-oss-20b` for generation; a separate high-capacity model (`openai/gpt-oss-120b`) serves as the independent RAGAS judge to prevent self-scoring bias.

---

## 🗺️ Roadmap

- [ ] Cross-encoder reranking on top of hybrid retrieval
- [ ] Migrate parent docstore from local file storage to managed Postgres
- [ ] Dockerized deployment
- [ ] CI pipeline (lint + test on push)
- [ ] Public hosted demo

See [`projectflow.md`](./projectflow.md) for the full build narrative, including the debugging story behind each of these decisions.

---

## 🤝 Contributing

This is currently a solo learning/portfolio project, but issues and suggestions are welcome — feel free to open an issue if you spot something worth improving.

---

## 📬 Contact

Built by **Sriman Soma** — feel free to connect on [www.linkedin.com/in/srimansoma](#) or reach out via email.

---

<p align="center"><i>⭐ If this project interests you, a star on the repo is always appreciated.</i></p>
