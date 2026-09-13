# 📄 Papeer — Research Paper RAG System

**Ask questions. Get grounded, cited answers straight from the research papers themselves — not the model's imagination.**

Papeer is a production-minded Retrieval-Augmented Generation (RAG) system purpose-built for interacting with academic research papers. It's designed around a pluggable architecture that runs entirely on free/local tooling for development, while being production-ready to swap in managed cloud models with zero application-logic changes.

---

## 🚀 Why Papeer

Most RAG demos stop at "upload a PDF, ask a question." Papeer goes further — it's built to actually answer correctly on dense, jargon-heavy research papers, and it's evaluated, not just demoed:

- ❌ **The naive version failed silently.** Early on, the retriever wasn't hallucinating — it was under-confident, frequently answering *"I don't have the answer"* even when the paper clearly had it.
- ✅ **Fixed with Parent Document Retrieval**, giving the LLM full contextual passages instead of narrow, disconnected chunks.
- 📊 **Evaluated with RAGAS** across 4 core dimensions: **0.93 Faithfulness**, **0.95 Answer Relevancy**, **0.94 Context Precision**, and **0.91 Context Recall**.
- 🔀 **Hybrid retrieval** (BM25 + dense embeddings + MMR) so exact terminology *and* conceptual questions both get answered well.

Read the full engineering story in [`PROJECT_FLOW.md`](./PROJECT_FLOW.md).

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Pluggable LLM Architecture** | Swap between local (Ollama) and cloud-hosted models via a factory pattern — zero changes to app logic |
| 📚 **Parent Document Retrieval** | Retrieves precise small chunks, answers with full parent context |
| 🔍 **Hybrid Search** | BM25 (sparse/keyword) + dense embeddings, re-ranked with MMR for diverse, non-redundant results |
| 🧵 **Stateless Chat Memory** | Per-session conversational context without cross-user state leakage |
| 📈 **RAGAS-Evaluated** | Quantitatively benchmarked retrieval & generation quality, not just vibes |
| ⚡ **LangGraph Orchestration** | The RAG pipeline is modeled as an explicit, inspectable graph rather than a black-box chain |
| 🖥️ **Streamlit UI** | Lightweight, fast interface for asking questions and reviewing answers |

---

## 🏗️ Architecture

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

**Design principle:** the `.env` config drives two independent factories — one for the LLM, one for embeddings — so the orchestration layer (`rag_graph.py`) and the UI never need to know or care whether inference is happening locally via Ollama or against a managed cloud API.

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| **Orchestration** | LangChain, LangGraph |
| **LLM (dev)** | Ollama — `qwen2.5:3b` (lightweight, local, rate-limit-free) |
| **Evaluation Judge** | Gemini (used only for RAGAS, since Ollama can't parallelize eval) |
| **Embeddings** | HuggingFace (`CacheBackedEmbeddings`) |
| **Vector Store** | Qdrant |
| **Retrieval** | Parent Document Retrieval + Hybrid Search (BM25 + Dense + MMR) |
| **Evaluation** | RAGAS (5-metric suite) |
| **Frontend** | Streamlit |
| **Environment** | Python 3, `venv` |

---

## 📁 Project Structure

```
papeer/
├── backend/
│   ├── __init__.py
│   ├── models.py         # Pydantic schemas / data models
│   ├── paper_loader.py    # Document ingestion & chunking
│   ├── vector_store.py    # Embeddings + Qdrant vector DB logic
│   ├── rag_graph.py        # LangGraph orchestration
│   └── btw_handler.py      # LLM handler layer
├── about_project.md        # Original design spec
├── PROJECT_FLOW.md         # Full engineering build log
├── requirements.txt
├── .env.example
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

### 3. Configure environment variables
Create a `.env` file in the root directory:
```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

### 4. Pull the local LLM (for development)
```bash
ollama pull qwen2.5:3b
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
 
> **Decoupled Judge Architecture**: To prevent self-evaluation bias and avoid rate-limit bottlenecks, evaluation is decoupled from pipeline generation. While the pipeline operates on the generator model (`openai/gpt-oss-20b` / local Ollama), evaluation scoring is judged independently using a distinct high-capacity model (`openai/gpt-oss-120b`).

---

## 🗺️ Roadmap

- [ ] Cross-encoder reranking on top of hybrid retrieval
- [ ] Migrate parent docstore from local file storage to managed Postgres
- [ ] Dockerized deployment
- [ ] CI pipeline (lint + test on push)
- [ ] Public hosted demo

See [`PROJECT_FLOW.md`](./PROJECT_FLOW.md) for the full build narrative, including the debugging story behind each of these decisions.

---

## 🤝 Contributing

This is currently a solo learning/portfolio project, but issues and suggestions are welcome — feel free to open an issue if you spot something worth improving.

---

## 📬 Contact

Built by **Sriman Soma** — feel free to connect on [www.linkedin.com/in/srimansoma](#) or reach out via email.

---

<p align="center"><i>⭐ If this project interests you, a star on the repo is always appreciated.</i></p>