"""
Backend/paper_loader.py
------------------------
Document ingestion layer — loads papers from multiple source types and returns
a flat list of LangChain Document objects ready for vector-store indexing.

Supported sources (dispatched automatically by `load_document`):
  - PDF files          (.pdf)  — via PyMuPDF
  - Plain-text files   (.txt)  — via LangChain TextLoader
  - Markdown files     (.md)   — via LangChain TextLoader
  - Web pages          (http/https URLs) — via LangChain WebBaseLoader
  - ArXiv papers       (ID like "1706.03762" or paper title string)

Every loaded document is stamped with a `title` metadata field so the
vector store and UI can display a human-readable source name.

Public API:
  load_document(source)   — auto-dispatches based on URL scheme or file extension
  load_arxiv(query)       — loads an arXiv paper by ID or title string
  load_pdf / load_text / load_markdown / load_webpage — individual loaders
"""

import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Constants ──────────────────────────────────────────────────────────────────

# Chunk settings used only for loaders that pre-split (text / markdown).
# The vector_store layer applies its own parent/child splitting on top.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Regex that matches a bare arXiv ID with an optional version suffix (e.g. 1706.03762v2)
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")


# ── Splitters ──────────────────────────────────────────────────────────────────

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)

_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _stamp_title(docs: list[Document], title: str) -> list[Document]:
    """Inject a 'title' key into every document's metadata in-place."""
    for doc in docs:
        doc.metadata["title"] = title
    return docs


def _extract_arxiv_id(query: str) -> str | None:
    """Return a bare arXiv ID (version suffix stripped) if one is found in *query*.

    Returns None if no ID pattern is present so the caller can fall back to a
    title-based search.
    """
    match = _ARXIV_ID_RE.search(query)
    if match:
        return re.sub(r"v\d+$", "", match.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch the paper title for *arxiv_id* from the arXiv Atom API.

    Falls back to the raw ID string if the title cannot be parsed.
    """
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        xml = resp.read().decode()
    titles = re.findall(r"<title>(.*?)</title>", xml, re.DOTALL)
    # The first <title> is the feed title; the second is the paper title.
    return titles[1].strip() if len(titles) > 1 else arxiv_id


def _arxiv_search(query: str) -> str:
    """Search the arXiv Atom API by title phrase and return the top result's bare ID.

    Raises:
        ValueError: If no matching paper is found.
    """
    phrase = query.strip('"')
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query={search_query}&max_results=1&sortBy=relevance"
    )
    with urllib.request.urlopen(url, timeout=15) as resp:
        xml = resp.read().decode()
    match = re.search(
        r"<id>https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)</id>", xml
    )
    if not match:
        raise ValueError(f"No arXiv paper found for query: {query!r}")
    return re.sub(r"v\d+$", "", match.group(1))


def _load_arxiv_by_id(arxiv_id: str) -> list[Document]:
    """Download the PDF for *arxiv_id* from arXiv, parse it, and return Documents.

    Downloads to a temporary file that is cleaned up on exit regardless of
    success or failure.

    Raises:
        ValueError: If the PDF cannot be loaded (empty result from PyMuPDF).
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        docs = PyMuPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError(f"Could not parse PDF for arXiv ID: {arxiv_id}")

        # Prefer the title embedded in the PDF metadata; fall back to API lookup.
        title = (docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        return _stamp_title(docs, title)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# ── Public loaders ─────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> list[Document]:
    """Load and parse a PDF file using PyMuPDF.

    Stamps the file stem (filename without extension) as the document title.
    """
    docs = PyMuPDFLoader(file_path).load()
    return _stamp_title(docs, Path(file_path).stem)


def load_text(file_path: str) -> list[Document]:
    """Load a plain-text (.txt) file.

    Stamps the file stem as the document title.
    """
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(docs, Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    """Load a Markdown (.md) file.

    Stamps the file stem as the document title.
    """
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _stamp_title(docs, Path(file_path).stem)


def load_webpage(url: str) -> list[Document]:
    """Load a web page via LangChain's WebBaseLoader.

    Uses the page's HTML <title> tag as the document title, falling back to
    the URL itself if no title is found.
    """
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _stamp_title(docs, title)


def load_arxiv(query: str) -> list[Document]:
    """Load an arXiv paper by ID or title string.

    If *query* contains a numeric arXiv ID (e.g. "1706.03762"), it is used
    directly.  Otherwise the arXiv search API is queried by title to find the
    best matching paper ID.

    Args:
        query: An arXiv paper ID (with or without version) or a title string.

    Returns:
        A list of Document objects parsed from the paper's PDF.
    """
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(arxiv_id)


def load_document(source: str) -> list[Document]:
    """Auto-dispatch loader based on URL scheme or file extension.

    Routes to:
      - load_webpage   for http:// / https:// sources
      - load_pdf       for .pdf files
      - load_text      for .txt files
      - load_markdown  for .md / .markdown files

    Args:
        source: A file path or URL string.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if source.startswith(("http://", "https://")):
        return load_webpage(source)

    ext = Path(source).suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_markdown,
        ".markdown": load_markdown,
    }
    if ext in loaders:
        return loaders[ext](source)

    raise ValueError(
        f"Unsupported file type: {ext!r}. "
        "Supported: .pdf, .txt, .md, .markdown, or an http(s) URL."
    )
