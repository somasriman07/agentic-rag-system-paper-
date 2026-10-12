"""
Backend/prompts.py
-------------------
All LLM prompt strings and prompt templates used across the RAG pipeline.

Keeping prompts in one place makes them easy to review, tune, and test
independently of the node logic that uses them.

Exports
-------
  ROUTER_SYSTEM_PROMPT        — intent classification for the router node
  RETRIEVE_SYSTEM_PROMPT      — tool-selection instructions for the retrieval agent
  QUERY_REWRITE_SYSTEM_PROMPT — query improvement after a failed retrieval
  CLAIM_ANALYSIS_PROMPT       — fact-checking instructions for the verification node
  RELEVANCY_CHECK_PROMPT      — ChatPromptTemplate for the relevancy gate node
"""

from langchain_core.prompts import ChatPromptTemplate


# ── Router ────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = (
    "You are an intelligent router for a research paper RAG system.\n"
    "Analyse the user's query and classify it into one of three routes:\n\n"
    "   - 'retrieve': Questions about research papers, methods, architectures, "
    "benchmark numbers, or any document-grounded query.\n"
    "   - 'verify_claim': Checking if a specific scientific claim/result is "
    "superseded or updated by newer literature.\n"
    "   - 'direct_answer': Pure conversational greetings, general knowledge, "
    "or basic non-paper questions.\n\n"
    "Also provide a concise 1-sentence reason for your choice."
)


# ── Retrieval agent ───────────────────────────────────────────────────────────

RETRIEVE_SYSTEM_PROMPT = (
    "You are a research assistant gathering context to answer a user's question "
    "about research papers.\n\n"
    "You have two tools available:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded paper collection.\n"
    "   Choose the query carefully to match relevant paper chunks; set k to the "
    "number of chunks needed (1–10).\n\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   Rewrite the user's question as a concise, keyword-rich web query.\n\n"
    "For questions about uploaded papers, always use retrieve_from_vectorstore.\n"
    "Do NOT produce a final answer — only call tools to collect context."
)


# ── Query rewrite ─────────────────────────────────────────────────────────────

QUERY_REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# ── Claim verification ────────────────────────────────────────────────────────

CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by "
    "more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update "
    "the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes "
    "the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty "
    "superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)


# ── Relevancy check ───────────────────────────────────────────────────────────

RELEVANCY_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You evaluate whether retrieved document chunks contain relevant information "
        "to answer the user's question about research papers.\n"
        "Be lenient: if there is any substantive overlap or relevant context, "
        "mark is_relevant as 'yes'.\n"
        "Return ONLY a JSON object with keys 'is_relevant' (yes/no) and 'reason'."
    ),
    ("human", "Question: {query}\n\nRetrieved Chunks:\n{chunks}\n\nJSON decision:"),
])
