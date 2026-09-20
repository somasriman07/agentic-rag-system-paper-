"""
Backend/vector_store.py
------------------------
Vector store layer — handles document indexing and hybrid retrieval.

Architecture
~~~~~~~~~~~~
Documents are stored using a *Parent Document Retrieval* strategy:
  - Small child chunks (500 tokens) are embedded and stored in Qdrant for
    precise semantic matching.
  - Full parent chunks (3 000 tokens) are stored in a local file-based
    docstore so the LLM receives rich context rather than narrow snippets.

Retrieval uses *Hybrid Search*:
  - Dense vector search  (Qdrant with MMR for diverse results)
  - Sparse keyword search (BM25 over child chunks)
  - Results are fused via Reciprocal Rank Fusion (EnsembleRetriever, 50/50 weight)
  - Matched child chunks are then mapped back to their parent documents.

Each chat session gets its own Qdrant collection and local docstore directory,
so documents from different conversations never bleed into each other.

Embedding cache
~~~~~~~~~~~~~~~
Embeddings are cached on disk under ./embedding_cache/ using LangChain's
CacheBackedEmbeddings with a BLAKE2b key encoder, so identical text is never
re-embedded across sessions or runs.

Public API
~~~~~~~~~~
  add_paper(docs, session_id)   — index a list of Documents for a session
  list_papers(session_id)       — return the titles of all indexed papers
  search(query, session_id, k)  — hybrid retrieval returning parent Documents
"""

import os
from dotenv import load_dotenv

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from Backend.embedding_factory import get_embedding_model
from Backend.config import EMBEDDING_DIM

load_dotenv(override=True)


# ── Chunk size configuration ──────────────────────────────────────────────────
# Parent chunks: large enough to give the LLM full context for an answer.
# Child chunks:  small enough for precise semantic matching in Qdrant.

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,  chunk_overlap=100)


# ── Embedding model + disk cache ──────────────────────────────────────────────
# The cache is keyed by (model_name, content_hash) so switching models
# automatically writes to a separate namespace.

_base_embeddings = get_embedding_model()
_embedding_file_store = LocalFileStore("./embedding_cache/")

# Resolve a stable namespace string from the underlying model object
_base_model_name: str = (
    getattr(_base_embeddings, "model_name", None)
    or getattr(_base_embeddings, "model", None)
    or _base_embeddings.__class__.__name__
)

# Public alias used by app.py to display the active embedding model in the UI
base_model_name: str = _base_model_name

embeddings = CacheBackedEmbeddings.from_bytes_store(
    _base_embeddings,
    _embedding_file_store,
    namespace=_base_model_name,
    query_embedding_cache=True,
    key_encoder="blake2b",
)


# ── Qdrant client (shared across all sessions) ────────────────────────────────

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Embedding dimension (lazy, cached after first probe) ──────────────────────

_embedding_dim: int | None = None


def get_embedding_dimension() -> int:
    """Return the vector dimension of the active embedding model.

    Result is cached after the first call to avoid repeated API/model round-trips.
    Falls back to the EMBEDDING_DIM env var if the probe fails.
    """
    global _embedding_dim
    if _embedding_dim is None:
        try:
            _embedding_dim = len(embeddings.embed_query("probe"))
        except Exception:
            _embedding_dim = EMBEDDING_DIM
    return _embedding_dim


# ── Collection helpers ────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    """Return the Qdrant collection name for *session_id*.

    Dashes are replaced with underscores because Qdrant collection names
    must not contain dashes.
    """
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    """Return (and auto-create) the Qdrant collection for *session_id*.

    If a collection already exists but its vector dimension does not match
    the current embedding model, it is recreated automatically (dimension
    mismatch can happen after switching EMBEDDING_PROVIDER).
    """
    collection_name = get_collection_name(session_id)
    dim = get_embedding_dimension()

    # Auto-heal stale collection if dimensions changed
    if qdrant_client.collection_exists(collection_name):
        try:
            info = qdrant_client.get_collection(collection_name)
            vectors_cfg = info.config.params.vectors
            existing_dim = (
                vectors_cfg.size
                if hasattr(vectors_cfg, "size")
                else vectors_cfg.get("size")
            )
            if existing_dim is not None and existing_dim != dim:
                qdrant_client.delete_collection(collection_name)
        except Exception:
            pass  # If inspection fails, proceed — creation will handle it

    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Parent document retriever helpers ─────────────────────────────────────────

def get_docstore(session_id: str):
    """Return the local file-based docstore for *session_id*.

    Parent documents are persisted under ./parent_document_store/<session_id>/
    so they survive server restarts.
    """
    fs = LocalFileStore(f"./parent_document_store/{session_id}")
    return create_kv_docstore(fs)


def get_parent_document_retriever(session_id: str) -> ParentDocumentRetriever:
    """Return a configured ParentDocumentRetriever for *session_id*.

    The retriever stores full parent chunks in the local docstore and embeds
    child chunks in Qdrant for fast similarity search.
    """
    return ParentDocumentRetriever(
        vectorstore=get_vectorstore(session_id),
        docstore=get_docstore(session_id),
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    """Index *docs* into the vector store and docstore for *session_id*.

    Child chunks are embedded and stored in Qdrant; full parent chunks are
    written to the local file docstore.

    Args:
        docs:       LangChain Document objects (e.g. from paper_loader).
        session_id: Unique identifier for the current chat session.
    """
    retriever = get_parent_document_retriever(session_id)
    retriever.add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    """Return the titles of all papers indexed for *session_id*.

    Scrolls the Qdrant collection in pages of 100 and de-duplicates by title.
    Returns an empty list if no collection exists yet.

    Args:
        session_id: Unique identifier for the current chat session.
    """
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []

    seen: set[str] = set()
    titles: list[str] = []
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break

    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Retrieve the top-*k* parent documents most relevant to *query*.

    Retrieval pipeline:
      1. Load all parent documents from the local docstore.
      2. Split them into child chunks to build an in-memory BM25 index.
      3. Run dense MMR retrieval from Qdrant.
      4. Fuse BM25 + dense results via Reciprocal Rank Fusion (50/50).
      5. Map each matched child chunk back to its parent document.
      6. Return de-duplicated parent documents (richer context for the LLM).

    Args:
        query:      The user's search query string.
        session_id: Unique identifier for the current chat session.
        k:          Number of results to retrieve from each retriever.

    Returns:
        A list of parent Document objects. Empty list if no documents are indexed.
    """
    docstore = get_docstore(session_id)
    vectorstore = get_vectorstore(session_id)

    # Load all parent documents for this session
    parent_docs: list[Document] = []
    for key in docstore.yield_keys():
        doc = docstore.mget([key])[0]
        if doc is not None:
            # Inject the docstore key so child chunks can trace back to their parent
            doc.metadata["doc_id"] = key
            parent_docs.append(doc)

    if not parent_docs:
        return []

    # Build an in-memory BM25 index over child chunks for keyword matching
    child_docs = child_splitter.split_documents(parent_docs)
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = k

    # Dense retriever with MMR to promote diverse, non-redundant results
    qdrant_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 2 * k},
    )

    # Fuse results via Reciprocal Rank Fusion
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[0.5, 0.5],
    )
    child_results = ensemble.invoke(query)

    # Map child chunks back to their parent documents (de-duplicated)
    seen_parent_ids: set[str] = set()
    retrieved_parents: list[Document] = []

    for child in child_results:
        parent_id = child.metadata.get("doc_id")
        if parent_id and parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parent_doc = docstore.mget([parent_id])[0]
            if parent_doc is not None:
                retrieved_parents.append(parent_doc)

    return retrieved_parents
