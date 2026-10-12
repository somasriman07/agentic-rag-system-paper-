"""
Backend/tools.py
-----------------
LangGraph tool definitions for the retrieval agent.

Two tools are registered:

  retrieve_from_vectorstore
    Searches the session's Qdrant collection using the hybrid retriever
    (BM25 + dense MMR + Reciprocal Rank Fusion). Returns parent document
    chunks for rich context. Results are accumulated across multiple tool
    calls within the same agent turn via a Command state update.

  web_search
    Searches the live web using Tavily. Used when the query needs
    information not present in the uploaded papers. Results are wrapped
    as LangChain Documents and accumulated in the same retrieved_docs list.

Both tools use InjectedState to read session_id and the current accumulated
docs from RAGState without requiring the agent to pass them explicitly.

Exports
-------
  retrieve_from_vectorstore  — @tool decorated function
  web_search                 — @tool decorated function
  RETRIEVAL_TOOLS            — list of both tools (passed to bind_tools)
"""

import logging
import os
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient

from Backend.vector_store import search as vs_search

logger = logging.getLogger(__name__)


# ── Input schemas ─────────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    """Input schema for the vector-store retrieval tool."""
    query: str = Field(description="Semantic query to search research paper chunks.")
    k:     int = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve.")


class WebSearchInput(BaseModel):
    """Input schema for the web search tool."""
    optimized_query: str = Field(description="Query rewritten and optimised for web search.")
    max_results:     int = Field(default=3, ge=1, le=10, description="Number of web results.")


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query:        str,
    k:            int,
    session_id:   Annotated[str,  InjectedState("session_id")],
    current_docs: Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id: Annotated[str,  InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages.

    Appends results to the accumulated retrieved_docs list in state so multiple
    tool calls within the same agent turn build up context incrementally.
    """
    try:
        docs = vs_search(query=query, session_id=session_id, k=k)
    except Exception as exc:
        logger.warning("Vector search failed (%s). Returning empty result.", exc)
        docs = []

    if not docs:
        return [ToolMessage(
            content="No relevant documents found in the vector store.",
            tool_call_id=tool_call_id,
        )]

    return [
        ToolMessage(
            content=f"Retrieved {len(docs)} chunk(s) from the vector store.",
            tool_call_id=tool_call_id,
        ),
        Command(update={"retrieved_docs": (current_docs or []) + docs}),
    ]


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results:     int,
    current_docs:    Annotated[list, InjectedState("retrieved_docs")],
    tool_call_id:    Annotated[str,  InjectedToolCallId],
) -> list:
    """Search the live web using Tavily and append results as Documents.

    Used when the query requires information not present in the uploaded papers,
    or when the router determines a web search is needed.
    """
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(optimized_query, max_results=max_results)

    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]

    web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ]
    return [
        ToolMessage(
            content=f"Found {len(web_docs)} web result(s) for: {optimized_query}",
            tool_call_id=tool_call_id,
        ),
        Command(update={"retrieved_docs": (current_docs or []) + web_docs}),
    ]


# ── Tool list ─────────────────────────────────────────────────────────────────

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
