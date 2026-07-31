╔══════════════════════════════════════════════════════════════════════════════╗
║                 🚀 AGENTIC RAG SYSTEM - DEVELOPMENT FLOW                    ║
║                    Production-Grade GenAI Project Journey                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

📌 PROJECT GOAL
──────────────────────────────────────────────────────────────────────────────
Build a production-grade Agentic Retrieval-Augmented Generation (RAG) system
that enables users to interact with research papers using intelligent agents,
vector search, web search, and LLM reasoning.

The project is designed following scalable software engineering principles so
that it can be easily extended, deployed, and integrated with different LLM
providers.


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 1 — PROJECT INITIALIZATION
═══════════════════════════════════════════════════════════════════════════════

✅ Created a dedicated GitHub repository

    • Initialized Git
    • Created remote repository
    • Cloned project locally
    • Established version control from Day 1

This allows every milestone to be tracked with meaningful commits instead of
one large final commit.


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 2 — DEVELOPMENT ENVIRONMENT
═══════════════════════════════════════════════════════════════════════════════

Created an isolated Python virtual environment.

MacOS

    python3 -m venv .venv
    source .venv/bin/activate

Installed all project dependencies

    pip install -r requirements.txt

Added

    .gitignore

to ignore

    • .venv/
    • __pycache__/
    • .env
    • compiled files
    • OS-specific files


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 3 — PROJECT STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

Designed a modular backend architecture instead of placing all code inside a
single file.

Backend/
│
├── __init__.py
├── paper_loader.py
├── vector_store.py
├── rag_graph.py
├── btw_handler.py
├── models.py
├── llm_factory.py
├── embedding_factory.py
└── config.py

Supporting Files

├── app.py
├── README.md
├── about_project.md
├── requirements.txt
├── .env
└── .gitignore

This structure improves

✔ Maintainability
✔ Readability
✔ Scalability
✔ Production readiness


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 4 — PROJECT DOCUMENTATION
═══════════════════════════════════════════════════════════════════════════════

Created

about_project.md

to clearly explain

• Project objective
• Features
• Technologies used
• Design decisions
• Recruiter-friendly project overview

This allows visitors to quickly understand the project before exploring the
codebase.


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 5 — DOCUMENT INGESTION PIPELINE
═══════════════════════════════════════════════════════════════════════════════

Implemented

paper_loader.py

Responsibilities

✔ Load research papers
✔ Extract text
✔ Split documents into chunks
✔ Preserve metadata
✔ Prepare documents for embedding generation

This module acts as the entry point of the RAG pipeline.


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 6 — VECTOR DATABASE
═══════════════════════════════════════════════════════════════════════════════

Implemented

vector_store.py

Environment Variables

QDRANT_URL=...
QDRANT_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...

Responsibilities

✔ Generate embeddings
✔ Store vectors inside Qdrant
✔ Retrieve relevant chunks
✔ Similarity Search
✔ Cache embeddings for faster retrieval

Embedding Cache

CacheBackedEmbeddings

Benefits

✔ Faster repeated indexing
✔ Lower embedding cost
✔ Reduced computation


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 7 — AGENT DECISION MODELS
═══════════════════════════════════════════════════════════════════════════════

Implemented

models.py

Contains

✔ Structured output models
✔ Routing decisions
✔ Validation schemas
✔ Agent communication objects

These models ensure reliable communication between different LangGraph nodes.


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 8 — BTW HANDLER (Off-topic Assistant)
═══════════════════════════════════════════════════════════════════════════════

Implemented

btw_handler.py

Purpose

Handle off-topic conversations without invoking the RAG pipeline.

Workflow

User Query
      │
      ▼
Need Web Search?
      │
 ┌────┴────┐
 │         │
 ▼         ▼
No        Yes
 │         │
 ▼         ▼
LLM     Tavily Search
 │         │
 └────┬────┘
      ▼
 Response

Model Used

Qwen2.5:3B (Ollama)

Reason for Selection

✔ Completely Free
✔ Runs Locally
✔ Lightweight
✔ Fast inference
✔ No API cost
✔ No rate limits
✔ Excellent for lightweight routing and conversational tasks

Heavy reasoning is delegated to the main Agentic RAG pipeline.


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 9 — PLUGGABLE AI ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Designed the system to support multiple LLM and embedding providers without
changing the application logic.

                      .env
                        │
                        ▼
      ┌────────────────────────────────────┐
      │                                    │
      ▼                                    ▼
 llm_factory.py                  embedding_factory.py
      │                                    │
      ▼                                    ▼
 rag_graph.py                    vector_store.py
      │                                    │
      ▼                                    ▼
 btw_handler.py               CacheBackedEmbeddings
      │                                    │
      └────────────────┬───────────────────┘
                       ▼
                  LangGraph Engine
                       │
                       ▼
                   Streamlit UI


Benefits

✔ Vendor Independent
✔ Easily switch between Ollama, OpenAI or Gemini
✔ Production Ready
✔ Environment-driven configuration
✔ Cleaner architecture
✔ Separation of concerns


═══════════════════════════════════════════════════════════════════════════════
🚩 PHASE 10 — STATE MANAGEMENT
═══════════════════════════════════════════════════════════════════════════════

Implemented

Stateless Chat Memory

Benefits

✔ Independent user sessions
✔ Easier scaling
✔ Cleaner conversation handling
✔ Production-friendly architecture


═══════════════════════════════════════════════════════════════════════════════
🚩 ENGINEERING PRINCIPLES FOLLOWED
═══════════════════════════════════════════════════════════════════════════════

✔ Modular Design
✔ Factory Pattern
✔ Environment-Based Configuration
✔ Pluggable LLM Architecture
✔ Pluggable Embedding Architecture
✔ Clean Folder Structure
✔ Separation of Concerns
✔ Production-Oriented Design
✔ Retrieval-Augmented Generation
✔ Caching for Performance
✔ Scalable Backend Components


═══════════════════════════════════════════════════════════════════════════════
🚩 DEVELOPMENT PHILOSOPHY
═══════════════════════════════════════════════════════════════════════════════

This project was intentionally built from scratch instead of cloning an
existing implementation.

The primary objective was not just to reproduce functionality, but to gain a
deep understanding of every stage of an Agentic RAG system—from document
ingestion and vector indexing to intelligent routing, retrieval, reasoning,
and response generation.

Each component was implemented incrementally with a focus on clean
architecture, modularity, maintainability, and production readiness, allowing
the system to evolve into a scalable and extensible GenAI application.

═══════════════════════════════════════════════════════════════════════════════