# 🚀 Agentic RAG Chatbot

### Intelligent Retrieval • Memory-Aware Conversations • Research Verification • Production-Oriented Architecture

> **A modern Agentic Retrieval-Augmented Generation (RAG) system built using an intelligent multi-step workflow that combines semantic search, web intelligence, query refinement, verification, and memory to deliver highly accurate and reliable responses.**

---

# 🎯 Project Objective

Design and develop an **Agentic RAG Chatbot** capable of answering questions from multiple knowledge sources while intelligently deciding **when to retrieve information, when to search the web, when to refine the query, and when to admit uncertainty**.

Unlike a traditional RAG pipeline, this project incorporates **memory, routing, verification, caching, iterative retrieval, and session management**, making it closer to production-grade AI assistants.

---

# ✨ Key Highlights

✅ Multi-Source Knowledge Ingestion

✅ Agentic Retrieval Workflow

✅ Multi-Session Memory

✅ Intelligent Query Rewriting

✅ Retrieval Quality Validation

✅ Claim Verification via Web Search

✅ Embedding Cache Optimization

✅ Automatic Session Naming

✅ Separate Knowledge Collections

✅ Privacy Mode (/btw)

---

# 📚 Supported Knowledge Sources

The chatbot can ingest information from multiple formats without requiring manual preprocessing.

### Supported Sources

* 📄 PDF Documents (.pdf)
* 📝 Markdown Files (.md)
* 📄 Text Files (.txt)
* 🌐 Website URLs
* 📖 arXiv Paper IDs
* 🔬 arXiv Paper Titles

This allows users to continuously expand the chatbot's knowledge base with research papers, documentation, notes, and web resources.

---

# 🧠 Short-Term Memory

The chatbot maintains conversational context throughout each session.

### Features

* Remembers previous messages
* Maintains conversational flow
* Understands follow-up questions
* Supports multiple independent chat sessions
* Context-aware responses

Every session behaves like an independent conversation with its own memory.

---

# 🗂️ Multi-Session Architecture

Unlike basic chatbots, this project supports **multiple isolated conversations**.

### Each Session Includes

* Independent chat history
* Separate memory
* Dedicated Vector Store collection
* Independent document indexing

This prevents knowledge leakage between conversations while improving retrieval precision.

---

# 🏷️ Automatic Session Naming

Instead of generic names like:

* Chat 1
* Chat 2

The chatbot automatically generates meaningful session titles by summarizing the conversation history.

Example:

```
Building Agentic RAG using LangGraph
```

```
Research on Vision Transformers
```

This makes long-term conversation management significantly easier.

---

# 🧭 Intelligent Router Node

The chatbot first decides **where the answer should come from** before generating a response.

```
                    User Query
                         │
                         ▼
                 Intelligent Router
                  /               \
                 /                 \
                ▼                   ▼
      Knowledge Base          Web Search
        (Qdrant)              (Internet)
```

The router dynamically selects the most appropriate retrieval strategy.

---

# ⚡ Qdrant Vector Database

The project uses **Qdrant** as its semantic vector database.

### Why Qdrant?

* ⚡ Extremely fast similarity search
* 📈 Built-in HNSW indexing
* 🎯 Built-in reranking support
* 🚀 High-performance vector retrieval
* 📚 Optimized for semantic search
* 🔍 Scalable vector collections

---

# 🔍 Intelligent Retrieval Pipeline

Instead of directly answering after retrieval, the chatbot evaluates the quality of retrieved documents.

```
User Question
      │
      ▼
Retrieve Documents
      │
      ▼
Relevance Check
      │
 ┌────┴────┐
 │         │
 │Yes      │No
 ▼         ▼
Generate   Rewrite Query
Answer         │
               ▼
         Retrieve Again
               │
          Maximum 2 Attempts
               │
               ▼
      Still Not Relevant?
               │
               ▼
      "I don't know the answer."
```

---

# ✅ Retrieval Relevance Checking

Every retrieved document is evaluated before answer generation.

### If Relevant

* Generate grounded response
* Reduce hallucinations
* Improve factual accuracy

### If Not Relevant

* Automatically rewrite the query
* Retry retrieval
* Maximum of **2 retrieval attempts**

If high-quality evidence still cannot be found, the chatbot safely returns:

> **"I don't know the answer."**

This fallback mechanism helps minimize hallucinated responses.

---

# 🌐 Research Verification Mode

One of the most powerful features of this project.

When the prompt includes instructions such as:

```
Verify this claim
```

or

```
Verify whether this research is still valid
```

the chatbot performs a web search and:

* Verifies factual correctness
* Detects outdated information
* Identifies newer research
* Suggests updated resources
* Recommends recent research papers
* Provides links or references for further reading

The newly discovered research papers can then be added back into the chatbot's knowledge base, enabling continuous knowledge expansion.

---

# 🔒 /btw Mode (Private Conversations)

Inspired by the idea behind **Claude Code**, this project introduces a privacy-focused command.

```
/btw
```

Questions asked using this command:

* Are **not stored** in chat history
* Do **not affect session memory**
* Are answered using:

  * Parametric LLM knowledge
  * Web Search (when required)

Ideal for temporary questions that users don't want saved.

---

# ⚡ Embedding Cache Optimization

Generating embeddings repeatedly for identical content is both expensive and slow.

To solve this, the project implements an **Embedding Cache**.

### Cache Storage Options

* Memory
* File Store
* Redis

### Current Implementation

✅ File Store

### Why File Store?

Since the chatbot is designed to run locally, a file-based cache provides an excellent balance of:

* Simplicity
* Performance
* Zero infrastructure overhead

### Why Not Redis?

Redis is more suitable for production environments managing **millions of embeddings (10M+)**, where distributed caching and high-throughput access are required.

### Benefits

* Avoids repeated embedding generation
* Reduces embedding API calls
* Lowers operational cost
* Improves response latency
* Speeds up document ingestion
* Reuses previously computed embeddings

---

# 🛠️ Intelligent Agentic Workflow

```
User Question
       │
       ▼
 Intelligent Router
       │
 ┌─────┴─────────────┐
 │                   │
 ▼                   ▼
Knowledge Base   Web Search
(Qdrant)
       │
       ▼
Retrieve Documents
       │
       ▼
Relevance Evaluation
       │
 ┌─────┴───────┐
 │             │
 ▼             ▼
Relevant    Rewrite Query
 │              │
 ▼              ▼
Generate   Retrieve Again
 │
 ▼
Final Answer
```

---

# 🚀 Core Capabilities

* Multi-format document ingestion
* Agentic routing architecture
* Semantic retrieval with Qdrant
* Built-in relevance validation
* Automatic query rewriting
* Intelligent fallback responses
* Multi-session conversation support
* Session-specific vector collections
* Short-term conversational memory
* Automatic session title generation
* Privacy mode using `/btw`
* Research claim verification
* Web-augmented fact checking
* Embedding cache optimization
* Production-oriented RAG workflow

---

# 💡 Why This Project Stands Out

Unlike a conventional Retrieval-Augmented Generation system, this chatbot integrates **memory, intelligent routing, iterative retrieval, document relevance validation, verification, privacy controls, session management, and embedding optimization** into a cohesive agentic workflow.

The result is a scalable, production-inspired AI assistant capable of delivering **more accurate, reliable, cost-efficient, and context-aware responses**, making it a strong demonstration of modern Generative AI engineering practices and an excellent portfolio project for AI/ML, GenAI, and LLM-focused roles.
