"""
Backend/btw_handler.py
-----------------------
Handler for the /btw side-channel command.

The /btw command lets users ask off-topic questions that are *not* stored in
the session's chat history and do *not* affect the LangGraph checkpointer state.
It is inspired by the concept of a private scratchpad — useful for quick
lookups that the user doesn't want mixed into the research conversation.

How it works
~~~~~~~~~~~~
1. A small LLM call classifies whether the question needs a live web search
   (e.g. current prices, recent news) or can be answered from general knowledge.
2. If a web search is needed, Tavily fetches the top-3 results and provides
   them as context in the answer prompt.
3. The answer is streamed back chunk-by-chunk so the UI can render it live.

This handler is intentionally stateless — it creates a fresh LLM instance on
each module import and never reads from or writes to any persistent store.

Public API
~~~~~~~~~~
  handle_btw(query) -> Generator[str, None, None]
"""

import os
from typing import Generator

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from Backend.models import BtwRouteDecision
from Backend.llm_factory import get_llm

load_dotenv(override=True)


def _get_btw_llm():
    """Return the LLM for /btw queries.

    Prefers Groq (fast, free, no daily cap) so /btw never burns the Gemini
    quota.  Falls back to the configured default provider if Groq is not set.
    """
    if os.getenv("GROQ_API_KEY", "").strip():
        try:
            return get_llm(provider="groq")
        except Exception:
            pass
    return get_llm()


# One shared LLM instance for the /btw handler
_llm = _get_btw_llm()


def _extract_text(chunk) -> str:
    """Safely extract a plain string from an LLM streaming chunk.

    Newer langchain-google-genai versions return content as a list of
    content-block dicts (e.g. [{"type": "text", "text": "..."}]) instead of
    a plain string.  This helper normalises both forms to a string.
    """
    content = chunk.content if hasattr(chunk, "content") else chunk
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content) if content else ""


def handle_btw(query: str) -> Generator[str, None, None]:
    """Answer a /btw query outside the main session context.

    Streams the response as string chunks so Streamlit can render text
    progressively without blocking.

    Flow:
      1. Classify whether the query needs a real-time web search.
      2. If yes: search Tavily, inject results as context.
      3. Stream the final answer through the appropriate prompt template.

    Args:
        query: The user's question, with the '/btw' prefix already stripped.

    Yields:
        String chunks of the LLM response as they arrive.
    """
    # ── Step 1: Decide if a web search is needed ───────────────────────────
    route_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Decide if answering this question requires a real-time web search "
            "(recent events, current prices, breaking news) or if your general "
            "knowledge is sufficient.",
        ),
        ("human", "{query}"),
    ])
    decision: BtwRouteDecision = (
        route_prompt | _llm.with_structured_output(BtwRouteDecision)
    ).invoke({"query": query})

    # ── Step 2: Build the answer prompt ────────────────────────────────────
    if decision.needs_web_search_bool:
        # Fetch live context from the web
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query, max_results=3)
        context = "\n\n".join(r["content"] for r in results["results"])
        sources = "\n".join(f"- {r['url']}" for r in results["results"])

        # Use pre-built messages instead of ChatPromptTemplate — web results
        # can contain curly braces (JSON, URLs with params) that LangChain's
        # f-string template parser incorrectly treats as template variables.
        from langchain_core.messages import SystemMessage as _SM, HumanMessage as _HM
        messages = [
            _SM(content=(
                "Answer the question using the web search results below. "
                "Be concise and cite your sources.\n\n"
                f"Results:\n{context}\n\nSources:\n{sources}"
            )),
            _HM(content=query),
        ]
        for chunk in _llm.stream(messages):
            text = _extract_text(chunk)
            if text:
                yield text
        return  # done — skip Step 3

    else:
        # Answer purely from model knowledge — no dynamic content, safe to template
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question concisely from your general knowledge."),
            ("human", "{query}"),
        ])

    # ── Step 3: Stream the response ────────────────────────────────────────
    for chunk in (answer_prompt | _llm).stream({"query": query}):
        text = _extract_text(chunk)
        if text:
            yield text
