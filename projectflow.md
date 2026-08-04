╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║         🚀 PRODUCTION-GRADE RESEARCH PAPER RAG ASSISTANT                               ║
║                                                                                        ║
║                  Project Development Journey | Engineering Workflow                    ║
║                                                                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

"Every production project starts with a simple idea.
 This file documents my complete engineering approach—from project setup
 to retrieval optimization and evaluation."

==============================================================================================
PHASE 1 ─ PROJECT INITIALIZATION
==============================================================================================

📌 Created Git Repository
    ├── Initialized local repository
    ├── Connected GitHub remote
    ├── Cloned project locally
    └── Started version controlled development

📌 Environment Setup

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

📌 Added essential project files

    ✓ .gitignore
    ✓ README.md
    ✓ requirements.txt
    ✓ LICENSE
    ✓ about_project.md      ← Custom designed project documentation


==============================================================================================
PHASE 2 ─ PROJECT STRUCTURE
==============================================================================================

backend/
│
├── __init__.py
├── paper_loader.py
├── vector_store.py
├── models.py
├── llm_factory.py
├── embedding_factory.py
├── rag_graph.py
├── btw_handler.py
└── ...

Goal:
Keep every component modular so individual pieces can be swapped without
changing the rest of the system.


==============================================================================================
PHASE 3 ─ DOCUMENT INGESTION PIPELINE
==============================================================================================

Research Paper
      │
      ▼
paper_loader.py
      │
      ▼
Document Cleaning
      │
      ▼
Chunking Strategy
      │
      ▼
Embedding Generation
      │
      ▼
Qdrant Vector Database
      │
      ▼
Ready for Retrieval

Implemented:

✓ PDF Loading
✓ Text preprocessing
✓ Intelligent chunking
✓ Metadata preservation
✓ Vector indexing


==============================================================================================
PHASE 4 ─ VECTOR DATABASE SETUP
==============================================================================================

Configured secure environment variables

.env

QDRANT_URL=...
QDRANT_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...

Responsibilities

vector_store.py

• Create collection
• Generate embeddings
• Store vectors
• Similarity search
• Metadata storage


==============================================================================================
PHASE 5 ─ PLUGGABLE MODEL ARCHITECTURE
==============================================================================================

One of the main engineering goals was to avoid hardcoding any LLM.

Instead, I designed a factory-based architecture.

                             .env
                               │
                               ▼
         ┌─────────────────────────────┐
         │                             │
         ▼                             ▼
 llm_factory.py               embedding_factory.py
         │                             │
         ▼                             ▼
     graph.py                  vector_store.py
         │                             │
         ▼                             ▼
   btw_handler.py           CacheBackedEmbeddings
         │                             │
         └──────────────┬──────────────┘
                        ▼
                    LangGraph
                        │
                        ▼
                   Streamlit UI


Benefits

✓ Easily switch models
✓ Production ready
✓ Cleaner architecture
✓ No application code changes required


==============================================================================================
PHASE 6 ─ LOCAL-FIRST DEVELOPMENT
==============================================================================================

Development LLM

Ollama
└── qwen2.5:3b

Reason

✓ Lightweight
✓ Fast local inference
✓ No API rate limits
✓ Offline development
✓ Lower development cost

Production Ready

The same architecture can switch to

• Gemini
• OpenAI
• Claude
• Azure OpenAI

without modifying business logic.


==============================================================================================
PHASE 7 ─ RAG PIPELINE
==============================================================================================

                 User Query
                      │
                      ▼
               Query Processing
                      │
                      ▼
          Embedding Generation
                      │
                      ▼
            Vector Similarity Search
                      │
                      ▼
        Retrieved Context Documents
                      │
                      ▼
              Prompt Construction
                      │
                      ▼
                  Selected LLM
                      │
                      ▼
               Final Response


==============================================================================================
PHASE 8 ─ LANGGRAPH ORCHESTRATION
==============================================================================================

Instead of creating a simple sequential pipeline,

I orchestrated the workflow using LangGraph.

Advantages

✓ Modular execution
✓ Easy debugging
✓ Future agent integration
✓ Production scalability
✓ Cleaner state management


==============================================================================================
PHASE 9 ─ CHAT MEMORY
==============================================================================================

Implemented

✓ Stateless Conversation Memory

Purpose

• Preserve previous interactions
• Better follow-up questions
• More natural conversations


==============================================================================================
PHASE 10 ─ RETRIEVAL TESTING
==============================================================================================

Initial Testing

Question
↓

Retriever

↓

❌ Wrong Context Retrieved

↓

LLM

↓

"I don't have enough information."

Observation

The LLM wasn't hallucinating.

Instead,

the retriever failed to retrieve the correct context.

Conclusion

The bottleneck was Retrieval,
NOT Generation.


==============================================================================================
PHASE 11 ─ RETRIEVAL OPTIMIZATION
==============================================================================================

Problem

Small chunks lost important surrounding context.

Solution

Implemented

Parent Document Retriever

Architecture

Question
    │
    ▼
Child Chunk Search
    │
    ▼
Retrieve Parent Document
    │
    ▼
Complete Context
    │
    ▼
LLM

Parent documents stored using Local File Store.

Result

✅ Better contextual retrieval

✅ Higher answer accuracy

✅ Significantly improved grounding

✅ Reduced retrieval failures

(Uff... finally the retriever started bringing the correct documents 😄)


==============================================================================================
PHASE 12 ─ EMBEDDING OPTIMIZATION
==============================================================================================

Implemented

✓ CacheBackedEmbeddings

Benefits

• Faster indexing
• Avoid repeated embedding generation
• Reduced API usage
• Improved development speed


==============================================================================================
PHASE 13 ─ EVALUATION
==============================================================================================

After the retrieval pipeline became stable,

I moved to systematic evaluation using RAGAS.

Metrics

✓ Faithfulness

✓ Answer Relevancy

✓ Context Precision

✓ Context Recall

✓ Response Relevancy

Observation

For evaluation,

I replaced the local Ollama model with Gemini because:

• Gemini supports faster API execution
• Better parallel evaluation
• More stable scoring
• Faster experimentation

The production pipeline remains model agnostic.


==============================================================================================
PHASE 14 ─ SOFTWARE ENGINEERING PRACTICES
==============================================================================================

Throughout development I followed

✓ Modular architecture

✓ Separation of concerns

✓ Environment variable management

✓ Factory Design Pattern

✓ Reusable components

✓ Configuration-driven design

✓ Git version control

✓ Incremental testing

✓ Retrieval-first debugging

✓ Evaluation-driven improvements


==============================================================================================
CURRENT ARCHITECTURE
==============================================================================================

                     Research Paper
                           │
                           ▼
                    Document Loader
                           │
                           ▼
                    Text Chunking
                           │
                           ▼
                  Embedding Factory
                           │
                           ▼
                CacheBackedEmbeddings
                           │
                           ▼
                  Qdrant Vector Store
                           │
                           ▼
               Parent Document Retriever
                           │
                           ▼
                     LangGraph Engine
                           │
                           ▼
                    Prompt Generation
                           │
                           ▼
                     LLM Factory
                           │
                           ▼
                  qwen2.5 / Gemini
                           │
                           ▼
                     Streamlit UI


==============================================================================================
WHAT THIS PROJECT TAUGHT ME
==============================================================================================

✔ Retrieval quality matters more than choosing a larger LLM.

✔ A modular architecture makes production migration effortless.

✔ Debugging RAG requires validating every stage independently:
   Loader → Chunking → Embeddings → Retrieval → Prompt → LLM.

✔ Evaluation should be data-driven, not intuition-driven.

✔ Building production AI systems is as much about software engineering
  as it is about machine learning.


==============================================================================================
NEXT IMPROVEMENTS
==============================================================================================

□ Hybrid Search (BM25 + Dense Retrieval)

□ Query Rewriting / HyDE

□ Cross Encoder Re-ranking

□ Multi-query Retrieval

□ Metadata Filtering

□ Streaming Responses

□ Conversation Summarization Memory

□ Observability (LangSmith / Phoenix)

□ Docker + CI/CD Deployment

□ Unit & Integration Tests

□ API Deployment (FastAPI)

□ Authentication & Rate Limiting


==============================================================================================
FINAL NOTE
==============================================================================================

This project was not built by simply connecting an LLM to a vector database.

It was developed iteratively—testing, identifying bottlenecks, optimizing
retrieval, evaluating with RAGAS, and designing a modular architecture that
can transition from local development to production with minimal code changes.

That engineering journey is what transformed this from a demo into a
production-oriented Retrieval-Augmented Generation (RAG) system.

                                           — Built with curiosity, debugging,
                                             and a lot of coffee ☕