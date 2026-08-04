import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from Backend.embedding_factory import get_embedding_model
from Backend.config import EMBEDDING_DIM
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv(override=True)

# ── Splitters ────────────────────────────────────────────────────────────────
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)




# ── Singletons ────────────────────────────────────────────────────────────────


base_embeddings = get_embedding_model()
embedding_file_store = LocalFileStore("./embedding_cache/")

# Get a safe model identifier name for the cache namespace
base_model_name = (
    getattr(base_embeddings, "model_name", None)
    or getattr(base_embeddings, "model", None)
    or base_embeddings.__class__.__name__
)

embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_model_name,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection ───────────────────────────────────────────────────────────────

_embedding_dim = None

def get_embedding_dimension() -> int:
    global _embedding_dim
    if _embedding_dim is None:
        try:
            _embedding_dim = len(embeddings.embed_query("probe"))
        except Exception:
            _embedding_dim = EMBEDDING_DIM
    return _embedding_dim


def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    dim = get_embedding_dimension()
    
    # Auto-heal: Recreate collection if dimension mismatch is detected
    if qdrant_client.collection_exists(collection_name):
        try:
            info = qdrant_client.get_collection(collection_name)
            existing_dim = None
            if hasattr(info.config.params.vectors, 'size'):
                existing_dim = info.config.params.vectors.size
            elif isinstance(info.config.params.vectors, dict) and 'size' in info.config.params.vectors:
                existing_dim = info.config.params.vectors['size']
            
            if existing_dim is not None and existing_dim != dim:
                qdrant_client.delete_collection(collection_name)
        except Exception:
            pass

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



# ── Helper Retrievers ─────────────────────────────────────────────────────────

def get_docstore(session_id: str):
    store_dir = f"./parent_document_store/{session_id}"
    fs = LocalFileStore(store_dir)
    return create_kv_docstore(fs)


def get_parent_document_retriever(session_id: str) -> ParentDocumentRetriever:
    vectorstore = get_vectorstore(session_id)
    docstore = get_docstore(session_id)
    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    retriever = get_parent_document_retriever(session_id)
    retriever.add_documents(docs)


def list_papers(session_id: str) -> list[str]:
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
    docstore = get_docstore(session_id)
    vectorstore = get_vectorstore(session_id)
    
    # Retrieve all parent documents currently loaded for this session
    keys = list(docstore.yield_keys())
    parent_docs = []
    for k_val in keys:
        doc = docstore.mget([k_val])[0]
        if doc is not None:
            # Inject the docstore key (doc_id) into parent metadata
            # so child chunks split from it inherit this reference
            doc.metadata["doc_id"] = k_val
            parent_docs.append(doc)
            
    if not parent_docs:
        return []
        
    # Split parent documents into child chunks to build local BM25 index
    child_docs = child_splitter.split_documents(parent_docs)
    
    # Setup lexical retriever
    bm25_retriever = BM25Retriever.from_documents(child_docs)
    bm25_retriever.k = k
    
    # Setup semantic vector retriever with MMR for diverse responses
    qdrant_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 2 * k}
    )
    
    # Combine via Reciprocal Rank Fusion (RRF)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, qdrant_retriever],
        weights=[0.5, 0.5]
    )
    
    # Retrieve blended child chunks
    child_results = ensemble_retriever.invoke(query)
    
    # Map retrieved child chunks back to their parent documents
    seen_parent_ids = set()
    retrieved_parents = []
    for child in child_results:
        parent_id = child.metadata.get("doc_id")
        if parent_id and parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parent_doc = docstore.mget([parent_id])[0]
            if parent_doc is not None:
                retrieved_parents.append(parent_doc)
                
    return retrieved_parents