╔════════════════════════════════════════════════════════════════════════════════════╗
║                    🚀 AGENTIC RAG SYSTEM — PROJECT DEVELOPMENT FLOW               ║
║                 Building a Production-Ready Agentic Retrieval System              ║
╚════════════════════════════════════════════════════════════════════════════════════╝

🎯 PROJECT OBJECTIVE
══════════════════════════════════════════════════════════════════════════════════════

Build a scalable, production-grade Agentic Retrieval-Augmented Generation (RAG)
application capable of understanding research papers through intelligent
retrieval, reasoning, web search, and LLM orchestration.

Core Goals
──────────
✔ Modular Architecture
✔ Production-Ready Backend
✔ Local & Cloud LLM Support
✔ Intelligent Agent Workflow
✔ Easy Deployment & Scalability


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 01 — PROJECT INITIALIZATION
══════════════════════════════════════════════════════════════════════════════════════

GitHub Repository
        │
        ▼
Initialize Git
        │
        ▼
Clone Repository
        │
        ▼
Version Control Ready

Achievements
────────────
✔ Created GitHub Repository
✔ Initialized Git
✔ Connected Remote Origin
✔ Established Clean Commit History


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 02 — DEVELOPMENT ENVIRONMENT
══════════════════════════════════════════════════════════════════════════════════════

MacOS Setup

python3 -m venv .venv
source .venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Ignored Files

.gitignore
│
├── .venv/
├── .env
├── __pycache__/
├── *.pyc
└── OS Files

Outcome
───────
✔ Isolated Development Environment
✔ Dependency Management
✔ Clean Repository


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 03 — PROJECT ARCHITECTURE
══════════════════════════════════════════════════════════════════════════════════════

Backend/
│
├── config.py
├── models.py
├── paper_loader.py
├── vector_store.py
├── rag_graph.py
├── btw_handler.py
├── llm_factory.py
└── embedding_factory.py

Application

├── app.py
├── README.md
├── about_project.md
├── requirements.txt
├── .env
└── .gitignore

Architecture Highlights
───────────────────────
✔ Modular Design
✔ Separation of Concerns
✔ Production-Oriented Structure
✔ Easily Extendable


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 04 — PROJECT DOCUMENTATION
══════════════════════════════════════════════════════════════════════════════════════

Created

about_project.md

Includes

✔ Project Overview
✔ System Architecture
✔ Features
✔ Technology Stack
✔ Design Decisions
✔ Recruiter-Friendly Documentation


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 05 — DOCUMENT INGESTION PIPELINE
══════════════════════════════════════════════════════════════════════════════════════

paper_loader.py

Workflow

Research Paper
      │
      ▼
Load PDF
      │
      ▼
Extract Text
      │
      ▼
Chunk Documents
      │
      ▼
Preserve Metadata
      │
      ▼
Ready for Embeddings

Responsibilities

✔ PDF Loading
✔ Text Extraction
✔ Chunking
✔ Metadata Processing


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 06 — VECTOR DATABASE
══════════════════════════════════════════════════════════════════════════════════════

vector_store.py

Pipeline

Documents
     │
     ▼
Embeddings
     │
     ▼
CacheBackedEmbeddings
     │
     ▼
Qdrant Vector DB
     │
     ▼
Similarity Search

Environment Variables

QDRANT_URL
QDRANT_API_KEY
HUGGINGFACEHUB_API_TOKEN

Features

✔ Embedding Generation
✔ Vector Storage
✔ Semantic Search
✔ Cached Embeddings
✔ Faster Retrieval


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 07 — AGENT DECISION MODELS
══════════════════════════════════════════════════════════════════════════════════════

models.py

Responsibilities

✔ Structured Outputs
✔ Routing Models
✔ Validation Schemas
✔ Agent Communication Objects

Result

Reliable interaction between LangGraph nodes.


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 08 — OFF-TOPIC ASSISTANT
══════════════════════════════════════════════════════════════════════════════════════

btw_handler.py

                    User Query
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
     Off Topic?                Research Query
            │                       │
            ▼                       ▼
      Local LLM               Agentic Pipeline

Model

Qwen2.5:3B (Ollama)

Advantages

✔ Runs Locally
✔ Free
✔ Fast
✔ Lightweight
✔ No API Cost
✔ No Rate Limits


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 09 — PLUGGABLE AI ARCHITECTURE
══════════════════════════════════════════════════════════════════════════════════════

                   .env
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
llm_factory.py            embedding_factory.py
      │                             │
      ▼                             ▼
rag_graph.py             vector_store.py
      │                             │
      └──────────────┬──────────────┘
                     ▼
               LangGraph Engine
                     │
                     ▼
                Streamlit UI

Supported Providers

✔ Ollama
✔ OpenAI
✔ Google Gemini

Benefits

✔ Vendor Independent
✔ Easily Switch Models
✔ Environment Driven
✔ Production Ready


══════════════════════════════════════════════════════════════════════════════════════
🟢 PHASE 10 — CHAT STATE MANAGEMENT
══════════════════════════════════════════════════════════════════════════════════════

Stateless Session Architecture

Benefits

✔ Independent Sessions
✔ Better Scalability
✔ Easier Deployment
✔ Cleaner Conversation Flow


══════════════════════════════════════════════════════════════════════════════════════
🏗 ENGINEERING PRINCIPLES
══════════════════════════════════════════════════════════════════════════════════════

✔ Clean Architecture
✔ Factory Design Pattern
✔ Modular Components
✔ Environment-Based Configuration
✔ Pluggable LLM Providers
✔ Pluggable Embedding Providers
✔ Retrieval-Augmented Generation (RAG)
✔ Semantic Search
✔ Caching Strategy
✔ Separation of Concerns
✔ Scalable Backend
✔ Production-Ready Design


══════════════════════════════════════════════════════════════════════════════════════
⚙ TECHNOLOGY STACK
══════════════════════════════════════════════════════════════════════════════════════

Frontend        → Streamlit
Backend         → Python
Framework       → LangGraph
LLM             → Ollama / OpenAI / Gemini
Embeddings      → HuggingFace
Vector DB       → Qdrant
Web Search      → Tavily
Cache           → CacheBackedEmbeddings
Environment     → python-dotenv
Version Control → Git & GitHub


══════════════════════════════════════════════════════════════════════════════════════
🎯 DEVELOPMENT PHILOSOPHY
══════════════════════════════════════════════════════════════════════════════════════

Rather than cloning an existing repository, this project was built incrementally
from scratch to understand every stage of an Agentic RAG system.

Each module was developed independently with emphasis on:

• Clean Architecture
• Software Engineering Best Practices
• Scalability
• Maintainability
• Extensibility
• Production Readiness

The resulting application is designed to be easily extendable with new LLMs,
embedding models, retrieval strategies, and deployment environments while
maintaining a clean and modular codebase.

══════════════════════════════════════════════════════════════════════════════════════