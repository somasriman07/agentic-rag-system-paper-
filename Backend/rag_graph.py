"""
Backend/rag_graph.py
---------------------
LangGraph orchestration — the core agentic RAG pipeline.

Graph overview
~~~~~~~~~~~~~~
Every user message enters the pipeline at the *router* node, which classifies
the query into an intent route and selects the appropriate model tier.  The
graph then branches into one of three paths:

  retrieve path:
    router → agent_node ↔ retrieval (tool calls) → relevancy_check
          → [query_rewrite → agent_node]* → generate_answer

  verify_claim path:
    router → verify_claim → generate_answer

  direct_answer path:
    router → generate_answer

Node responsibilities
~~~~~~~~~~~~~~~~~~~~~
  router_node          — dual intent + model-tier classification
  agent_node           — runs the retrieval agent (decides which tools to call)
  retrieval            — ToolNode that executes retrieve_from_vectorstore / web_search
  relevancy_check_node — judges whether retrieved chunks are good enough to answer
  query_rewrite_node   — rewrites the query when retrieval quality is poor (max 1 retry)
  verify_claim_node    — web + arXiv search to check if a claim is still valid
  generate_answer_node — synthesises the final answer using the routed LLM tier

State
~~~~~
RAGState extends MessagesState (LangGraph's built-in message list) with
domain-specific fields tracked across the graph run.

Persistence
~~~~~~~~~~~
Graph state is persisted to a SQLite database (checkpoints.db) via LangGraph's
SqliteSaver.  Each chat session maps to a unique thread_id so conversations are
fully independent and survive server restarts.
"""

import json
import logging
import os
import re
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient
import valkey.asyncio as valkey_async
from betterdb_semantic_cache import SemanticCache, SemanticCacheOptions
from betterdb_semantic_cache.embed.openai import create_openai_embed
from betterdb_semantic_cache.types import CacheStoreOptions

from Backend.llm_factory import get_llm, get_groq_llm
from Backend.models import ClaimVerificationResult, DualRouterDecision
from Backend.vector_store import search as vs_search
from Backend.semantic_cache import cache_check, cache_store

load_dotenv(override=True)
logger = logging.getLogger(__name__)


# ── LLM singleton ────────────────────────────────────────────────────────────
# Single Groq model (openai/gpt-oss-20b) handles all pipeline tasks:
# routing, retrieval agent, relevancy check, query rewrite, claim verification,
# and final answer generation.

llm = get_groq_llm()


# ── Graph state ───────────────────────────────────────────────────────────────

class RAGState(MessagesState):
    """Full state carried through the LangGraph pipeline.

    Inherits the `messages` list from MessagesState.  All other fields
    are RAG-specific and are reset / updated by individual nodes.

    Fields:
        session_id          Unique chat session identifier (maps to a Qdrant collection).
        query               The user's original query text (preserved across rewrites).
        route               Intent route set by the router: 'retrieve' | 'verify_claim' | 'direct_answer'.
        route_reason        One-sentence rationale for the route decision (shown in UI).
        retrieved_docs      Documents accumulated by the retrieval agent across tool calls.
        retrieval_attempts  Number of times the retrieval agent has called a tool.
        claim_verdict       Verdict summary from the claim-verification node.
        claim_source        URL of the first superseding paper (if any).
        superseding_papers  List of dicts describing papers that supersede the claim.
        answer              Final generated answer text.
        is_relevant         Whether retrieved docs passed the relevancy gate (None = not yet checked).
        rewrite_count       Number of query rewrites performed (capped at 1).
    """

    session_id:          str
    query:               str
    route:               str | None
    route_reason:        str | None
    retrieved_docs:      list[Document]
    retrieval_attempts:  int
    claim_verdict:       str | None
    claim_source:        str | None
    superseding_papers:  list[dict] | None
    answer:              str | None
    is_relevant:         bool | None
    rewrite_count:       int
    # ── Semantic cache fields ──────────────────────────────────────────────
    cache_hit:           bool | None
    cache_similarity:    float | None
    cache_cost_saved:    float | None


# ── Prompt constants ──────────────────────────────────────────────────────────
# All prompts live here so they can be reviewed, tuned, or unit-tested without
# digging through node functions.

# Router system prompt — guides the LLM to classify intent and model tier
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

# Retrieval agent system prompt — instructs the agent on tool selection
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

# Query rewrite system prompt — asks for a better search query after failure
QUERY_REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)

# Claim analysis prompt — used inside verify_claim_node
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

# Relevancy check prompt template
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


# ── Pipeline constants ─────────────────────────────────────────────────────────

# Maximum number of tool-call rounds the retrieval agent may perform per query
MAX_RETRIEVAL_ATTEMPTS = 3


# ── Tool input schemas ────────────────────────────────────────────────────────

class RetrieverInput(BaseModel):
    """Input schema for the vector-store retrieval tool."""
    query: str  = Field(description="Semantic query to search research paper chunks.")
    k:     int  = Field(default=4, ge=1, le=10, description="Number of chunks to retrieve.")


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


# ── Retrieval agent setup ─────────────────────────────────────────────────────
# The retrieval agent is the frontier LLM bound to the tools above.  It decides
# autonomously which tool(s) to call and with what arguments.

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]

# Gemini does not support parallel tool calls in all versions; disable them
# for OpenAI-compatible backends to avoid "multiple tool calls" errors.
if type(llm).__name__ == "ChatOpenAI":
    retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
else:
    retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS)

base_tool_node = ToolNode(RETRIEVAL_TOOLS)

# Structured-output chains used by router and verification nodes
router_structured_llm  = llm.with_structured_output(DualRouterDecision)
verification_llm       = llm.with_structured_output(ClaimVerificationResult)
relevancy_chain        = RELEVANCY_CHECK_PROMPT | llm


# ── Graph nodes ───────────────────────────────────────────────────────────────

def router_node(state: RAGState) -> dict:
    """Classify the query and choose the intent route.

    Fast-path: obvious conversational queries are classified locally with no
    LLM call at all, saving a Groq round-trip for simple greetings.

    For everything else, calls Groq with a structured output schema to produce
    a DualRouterDecision.  Falls back to keyword heuristics if the LLM call
    fails — the user always gets a response, never an error.
    """
    query = state["messages"][-1].content

    # ── Fast-path: detect conversational queries without an LLM call ──────────
    _CONVERSATIONAL_PATTERNS = [
        "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
        "how are you", "who are you", "what are you", "what can you do",
        "my name is", "i am ", "i'm ", "nice to meet",
        "thanks", "thank you", "bye", "goodbye",
    ]
    _RESEARCH_KEYWORDS = [
        "paper", "research", "study", "model", "dataset", "method", "approach",
        "algorithm", "accuracy", "benchmark", "result", "experiment", "figure",
        "table", "section", "abstract", "conclusion", "architecture", "training",
        "inference", "performance", "compare", "verify", "claim", "arxiv",
    ]
    q_lower = query.lower().strip()
    is_conversational = (
        len(query.split()) <= 12
        and any(pat in q_lower for pat in _CONVERSATIONAL_PATTERNS)
        and not any(kw in q_lower for kw in _RESEARCH_KEYWORDS)
    )
    if is_conversational:
        return {
            "route":        "direct_answer",
            "route_reason": "Conversational query — answered directly without an LLM router call.",
        }

    # ── Default values used if LLM call and heuristics both fail ──────────────
    route  = "retrieve"
    reason = "Factual research retrieval."

    try:
        decision: DualRouterDecision = router_structured_llm.invoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
        route  = decision.route
        reason = decision.reason

    except Exception as exc:
        # LLM router failed — apply keyword heuristics immediately
        logger.warning("Router LLM failed (%s). Applying heuristic fallback.", exc)
        q = query.lower()

        if any(t in q for t in ["verify", "is it true", "superseded", "still valid"]):
            route  = "verify_claim"
            reason = "Scientific claim verification."
        elif any(t in q for t in ["what is your", "tell me about yourself",
                                   "how do you work", "what can you do",
                                   "who made you", "architecture of this",
                                   "how does this work"]):
            route  = "direct_answer"
            reason = "General question about the assistant."
        else:
            route  = "retrieve"
            reason = "Heuristic fallback: routed to retrieval."

    return {
        "route":        route,
        "route_reason": reason,
    }


def agent_node(state: RAGState) -> dict:
    """Run one round of the retrieval agent.

    Invokes the LLM with the current message history and RETRIEVE_SYSTEM_PROMPT.
    If the model returns tool calls they are executed by the downstream
    'retrieval' ToolNode and the attempt count is incremented.
    Once MAX_RETRIEVAL_ATTEMPTS is reached this node returns an empty dict
    to let the graph route to relevancy_check / generate_answer.
    """
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return {}

    messages  = [{"role": "system", "content": RETRIEVE_SYSTEM_PROMPT}] + state["messages"]
    response  = retrieval_llm.invoke(messages)
    updates: dict = {"messages": [response]}

    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1

    return updates


def relevancy_check_node(state: RAGState) -> dict:
    """Judge whether retrieved document chunks are relevant to the query.

    Uses the frontier LLM to assess the top-3 retrieved chunks against the
    query and returns a boolean `is_relevant` value.  Defaults to True if the
    LLM call fails so retrieval errors do not silently block answer generation.
    """
    query        = state["query"]
    docs         = state.get("retrieved_docs") or []
    doc_snippets = "\n\n---\n\n".join(doc.page_content for doc in docs[:3])

    if not doc_snippets:
        return {"is_relevant": False}

    try:
        response = relevancy_chain.invoke({"query": query, "chunks": doc_snippets})
        content  = str(
            response.content if hasattr(response, "content") else response
        ).strip()

        # Parse the JSON decision from the LLM response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            data   = json.loads(match.group(0))
            is_rel = str(data.get("is_relevant", "")).strip().lower() in ("yes", "true", "1")
            return {"is_relevant": is_rel}

        # Fallback: plain-text heuristic
        return {"is_relevant": "yes" in content.lower() or "true" in content.lower()}

    except Exception:
        # Default to relevant to avoid blocking the pipeline on errors
        return {"is_relevant": True}


def query_rewrite_node(state: RAGState) -> dict:
    """Rewrite the query when initial retrieval did not produce relevant results.

    Calls the LLM with the QUERY_REWRITE_SYSTEM_PROMPT to generate a better
    search query, then resets retrieval state so the agent tries again from
    scratch with the new query.
    """
    response = llm.invoke([
        {"role": "system",  "content": QUERY_REWRITE_SYSTEM_PROMPT},
        {"role": "user",    "content": f"Original query: {state['query']}\n\nWrite an improved search query."},
    ])
    rewritten = response.content.strip()

    return {
        "messages":           [HumanMessage(content=rewritten)],
        "query":              rewritten,
        "retrieved_docs":     [],
        "retrieval_attempts": 0,
        "rewrite_count":      state.get("rewrite_count", 0) + 1,
        "is_relevant":        None,
    }


def verify_claim_node(state: RAGState) -> dict:
    """Check whether a research claim has been superseded by newer work.

    Performs two Tavily searches:
      1. General web search for recent papers contradicting the claim.
      2. arXiv-targeted search for academic papers on the same topic.

    Results are passed to the verification LLM which returns a structured
    ClaimVerificationResult with a verdict and up to 3 superseding papers.
    """
    claim          = state["messages"][-1].content
    tavily_client  = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # Search 1: general web — broad coverage of recent challenges to the claim
    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # Search 2: arXiv-targeted — higher chance of finding peer-reviewed papers
    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # Build a structured context block for the verification LLM
    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )
    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{chr(10).join(lines)}"
    )

    result: ClaimVerificationResult = verification_llm.invoke([
        {"role": "user", "content": prompt}
    ])

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict":      result.verdict_summary,
        "claim_source":       papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def generate_answer_node(state: RAGState) -> dict:
    """Synthesise the final answer using Groq.

    Semantic cache check runs first — if a semantically similar query was
    answered before (cosine similarity ≥ SEMANTIC_CACHE_THRESHOLD), the
    cached answer is returned immediately with no LLM call.

    On a cache miss the answer is generated by Groq and stored in the cache
    so future similar queries benefit from it.

    Handles three answer paths:
      'retrieve'      — uses retrieved document context
      'verify_claim'  — pre-formatted from claim verdict (no LLM call)
      'direct_answer' — uses conversation history only
    """
    route = state.get("route")
    query = state["query"]

    # ── Semantic cache check ───────────────────────────────────────────────
    _cache_eligible = route in ("retrieve", "direct_answer")
    if _cache_eligible:
        result = cache_check(query)
        if result is not None and result.hit and result.response:
            logger.info(
                "Semantic cache HIT  query=%r  similarity=%.3f  cost_saved=$%.4f",
                query[:60], result.similarity or 0.0, result.cost_saved or 0.0,
            )
            clean_cached = re.sub(
                r"<think>[\s\S]*?</think>", "", str(result.response)
            ).strip()
            return {
                "answer":           clean_cached,
                "messages":         [AIMessage(content=clean_cached)],
                "cache_hit":        True,
                "cache_similarity": result.similarity,
                "cache_cost_saved": result.cost_saved,
            }

    # ── Path 1: Document retrieval answer ─────────────────────────────────
    if route == "retrieve":
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= 1:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. Try rephrasing your question or uploading "
                "additional papers."
            )
        else:
            docs = state.get("retrieved_docs") or []
            if not docs:
                answer = "I don't have enough information in the provided documents to answer this."
            else:
                context = "\n\n---\n\n".join(doc.page_content[:1500] for doc in docs[:3])
                system_message = SystemMessage(
                    content=(
                        "You are a helpful research assistant. Answer the user's question "
                        "concisely using the retrieved document context and conversation history.\n\n"
                        f"Retrieved Context:\n{context}"
                    )
                )
                messages = [system_message] + state["messages"]
                answer = llm.invoke(messages).content

    # ── Path 2: Claim verification answer (pre-formatted, no LLM call) ────
    elif route == "verify_claim":
        verdict    = state.get("claim_verdict", "")
        papers     = state.get("superseding_papers") or []
        claim_text = state["query"]

        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    # ── Path 3: Direct answer (general knowledge) ─────────────────────────
    else:
        messages = [
            SystemMessage(
                content="You are a helpful research assistant. Answer the user's question "
                        "using the conversation history and your knowledge."
            )
        ] + state["messages"]
        answer = llm.invoke(messages).content

    # ── Normalise answer to a plain string ─────────────────────────────────
    if isinstance(answer, list):
        answer = "".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in answer
        )

    clean_answer = re.sub(r"<think>[\s\S]*?</think>", "", str(answer)).strip()

    # ── Semantic cache store ───────────────────────────────────────────────
    if _cache_eligible:
        cache_store(query, clean_answer)
        logger.info("Semantic cache MISS — answer stored for query=%r", query[:60])

    return {
        "answer":           clean_answer,
        "messages":         [AIMessage(content=clean_answer)],
        "cache_hit":        False,
        "cache_similarity": None,
        "cache_cost_saved": None,
    }


# ── Routing functions ─────────────────────────────────────────────────────────
# These are pure functions that read state and return a string edge label.
# They contain no LLM calls.

def route_query(state: RAGState) -> str:
    """Edge function: branch on the intent route set by router_node."""
    return state["route"]


def agent_routing(state: RAGState) -> str:
    """Edge function: decide what follows after agent_node runs.

    Priority order:
      1. If there are pending tool calls → execute them ('retrieval')
      2. If retrieval attempts are exhausted → skip check and generate answer
      3. Otherwise → run the relevancy gate
    """
    if tools_condition(state) == "tools":
        # Always honour pending tool calls to keep the message history consistent
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def after_relevancy_routing(state: RAGState) -> str:
    """Edge function: decide what follows after relevancy_check_node.

    - Relevant docs → generate the answer
    - Irrelevant + rewrite budget remaining → rewrite the query and retry
    - Irrelevant + no budget → generate a graceful fallback answer
    """
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < 1:
        return "query_rewrite"
    return "generate_answer"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(db_path: str = "checkpoints.db"):
    """Build, compile, and return the LangGraph RAG pipeline.

    Creates a SQLite-backed checkpointer so conversation state persists across
    server restarts.  Each thread_id (= session_id) has isolated state.

    Args:
        db_path: Path to the SQLite database file for checkpointing.

    Returns:
        A compiled LangGraph StateGraph ready to call `.stream()` or `.invoke()` on.
    """
    conn        = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)

    # ── Register nodes ─────────────────────────────────────────────────────
    graph.add_node("router",          router_node)
    graph.add_node("agent_node",      agent_node)
    graph.add_node("retrieval",       base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite",   query_rewrite_node)
    graph.add_node("verify_claim",    verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── Edges ──────────────────────────────────────────────────────────────

    # 1. Router → intent branch
    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "retrieve":      "agent_node",
            "verify_claim":  "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    # 2. Retrieval agent loop
    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval":       "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieval", "agent_node")   # tool results feed back to agent

    # 3. Relevancy gate → rewrite or generate
    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {
            "query_rewrite":   "query_rewrite",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("query_rewrite", "agent_node")  # retry retrieval after rewrite

    # 4. Claim verification → answer generation
    graph.add_edge("verify_claim",    "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=checkpointer)
