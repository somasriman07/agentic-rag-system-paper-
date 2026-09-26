"""
Backend/llm_factory.py
-----------------------
Factory for chat LLM clients.

Centralises provider selection so every other module (rag_graph, btw_handler,
app.py) imports a model through one of the public functions below rather than
instantiating a concrete LangChain class directly.  Swapping providers is a
single env-var change — no application logic changes needed.

Supported providers (set via LLM_PROVIDER):
  gemini      — Google Gemini via langchain_google_genai  (default)
  ollama      — Local Ollama server
  vllm        — OpenAI-compatible vLLM inference server
  openai      — OpenAI API
  groq        — Groq hosted inference

Rate-limit resilience (Gemini free tier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Gemini free tier enforces a hard daily request quota.  When that quota is
hit, the API returns an HTTP 429 / ResourceExhausted error with a recommended
retry_delay in the response body.

`_RateLimitedGemini` is a transparent proxy that wraps the real
`ChatGoogleGenerativeAI` instance and retries *every* call that triggers a
quota error.  It:
  - Reads the `retry_delay { seconds: N }` field from the error message so the
    exact wait window requested by the API is respected.
  - Falls back to exponential back-off (base * 2^attempt) if no delay hint is
    present.
  - Delegates *all* other attribute accesses to the underlying model so it is a
    drop-in replacement throughout the codebase.

Relevant env vars:
  LLM_PROVIDER          ollama | vllm | openai | gemini | groq   (default: gemini)
  GEMINI_MODEL          Gemini model name                         (default: gemini-3.6-flash)
  GEMINI_API_KEY        Required for Gemini
  GEMINI_MAX_RETRIES    Max retries on 429 before giving up       (default: 6)
  GEMINI_INITIAL_WAIT   Base wait seconds for exponential backoff (default: 10)
  OPENWEIGHT_PROVIDER   ollama | vllm                             (default: ollama)
  OLLAMA_MODEL          Ollama model tag                          (default: qwen2.5:3b)
  OLLAMA_BASE_URL       Ollama server base URL                    (default: http://localhost:11434)
  VLLM_BASE_URL         vLLM OpenAI-compat endpoint               (default: http://localhost:8000/v1)
  VLLM_MODEL            vLLM model identifier                     (default: Qwen/Qwen2.5-7B-Instruct)
  VLLM_API_KEY          vLLM API key placeholder                  (default: EMPTY)
  GROQ_MODEL            Groq model identifier                     (default: openai/gpt-oss-20b)
  GROQ_API_KEY          Required for Groq

Public API:
  get_llm(provider)       — return any supported provider's model
  get_frontier_llm()      — shortcut: Gemini Frontier model
  get_openweight_llm()    — shortcut: Open-Weight model (Ollama / vLLM), with Groq fallback
  get_llm_by_tier(tier)   — resolve "gemini"/"frontier" or "qwen"/"openweight" string to model
"""

import os
import re
import time
import logging
import functools
from typing import Any, Iterator, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.runnables import RunnableSerializable

load_dotenv(override=True)
logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("gemini", "ollama", "vllm", "openai", "groq")


# ── Rate-limit retry decorator ────────────────────────────────────────────────

def _retry_on_resource_exhausted(func):
    """Decorator: retry *func* with back-off on Gemini free-tier 429 errors.

    Reads the `retry_delay { seconds: N }` value from the error message when
    available so the exact API-recommended wait is honoured.  Falls back to
    exponential back-off (GEMINI_INITIAL_WAIT * 2^attempt) otherwise.

    Controlled by env vars:
      GEMINI_MAX_RETRIES   (default: 6)
      GEMINI_INITIAL_WAIT  (default: 10 seconds)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "6"))
        base_wait   = float(os.getenv("GEMINI_INITIAL_WAIT", "10"))

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                exc_str = str(exc)
                # Only retry on quota / rate-limit errors
                is_quota = (
                    "429" in exc_str
                    or "ResourceExhausted" in exc_str
                    or "RESOURCE_EXHAUSTED" in exc_str
                    or "quota" in exc_str.lower()
                )
                if not is_quota or attempt >= max_retries:
                    raise  # Non-quota error or retries exhausted — propagate

                # Honour the API's suggested retry delay when present
                delay_match = re.search(
                    r"retry[_\s]delay\s*\{[^}]*seconds:\s*(\d+)", exc_str
                )
                wait = (
                    float(delay_match.group(1)) + 2  # +2s safety buffer
                    if delay_match
                    else base_wait * (2 ** attempt)  # exponential back-off
                )
                logger.warning(
                    "Gemini 429 ResourceExhausted (attempt %d/%d). Retrying in %.1fs …",
                    attempt + 1, max_retries, wait,
                )
                time.sleep(wait)

    return wrapper


# ── Rate-limited Gemini proxy ─────────────────────────────────────────────────

class _RateLimitedGemini(BaseChatModel):
    """Transparent proxy around ChatGoogleGenerativeAI that auto-retries on 429.

    Subclasses BaseChatModel so LangChain's pipe operator (prompt | llm) and
    type checks accept it as a proper Runnable.  Every LangChain method call
    (invoke, stream, batch, with_structured_output) is intercepted and wrapped
    with the retry decorator.  All other attribute accesses are forwarded to the
    underlying model unchanged.
    """

    # Pydantic field — holds the wrapped model
    _model: Any = None

    def __init__(self, model: BaseChatModel, **kwargs) -> None:
        # Bypass Pydantic's __init__ for the proxy — store model directly
        object.__setattr__(self, "_model", model)
        # Don't call super().__init__() — we're a proxy, not a real BaseChatModel

    # ── Attribute delegation ───────────────────────────────────────────────

    def __getattr__(self, name: str):
        """Forward unknown attribute lookups to the wrapped model."""
        model = object.__getattribute__(self, "_model")
        return getattr(model, name)

    def __setattr__(self, name: str, value) -> None:
        if name == "_model":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_model"), name, value)

    # ── Required BaseChatModel abstract methods ────────────────────────────
    # These delegate to the real model so Pydantic / LangChain internals work.

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return object.__getattribute__(self, "_model")._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    @property
    def _llm_type(self) -> str:
        return getattr(object.__getattribute__(self, "_model"), "_llm_type", "rate_limited_gemini")

    # ── Retried LangChain core methods ─────────────────────────────────────

    @_retry_on_resource_exhausted
    def invoke(self, *args, **kwargs):
        """Invoke the model with automatic 429 retry."""
        return object.__getattribute__(self, "_model").invoke(*args, **kwargs)

    @_retry_on_resource_exhausted
    def stream(self, *args, **kwargs):
        """Stream a response with automatic 429 retry."""
        return object.__getattribute__(self, "_model").stream(*args, **kwargs)

    @_retry_on_resource_exhausted
    def batch(self, *args, **kwargs):
        """Batch invoke with automatic 429 retry."""
        return object.__getattribute__(self, "_model").batch(*args, **kwargs)

    # ── LangChain pipe-chaining support ────────────────────────────────────
    # Delegate __or__ / __ror__ to the real model so (prompt | llm) works.

    def __or__(self, other):
        return object.__getattribute__(self, "_model").__or__(other)

    def __ror__(self, other):
        return object.__getattribute__(self, "_model").__ror__(other)

    # ── Structured output ──────────────────────────────────────────────────

    def with_structured_output(self, *args, **kwargs):
        """Return a retry-wrapped structured-output runnable.

        Delegates to the underlying model's with_structured_output(), then
        wraps the result in _RateLimitedRunnable so its invoke() also retries.
        """
        inner = object.__getattribute__(self, "_model").with_structured_output(
            *args, **kwargs
        )
        return _RateLimitedRunnable(inner)

    def bind_tools(self, *args, **kwargs):
        """Delegate bind_tools to the real model (used by the retrieval agent)."""
        return object.__getattribute__(self, "_model").bind_tools(*args, **kwargs)

    def __repr__(self) -> str:
        return f"_RateLimitedGemini({object.__getattribute__(self, '_model')!r})"


class _RateLimitedRunnable(RunnableSerializable):
    """Retry wrapper for arbitrary LangChain Runnables (e.g. structured-output chains).

    Subclasses RunnableSerializable so LangChain's pipe operator (prompt | runnable)
    accepts it as a valid Runnable.  Used internally by
    _RateLimitedGemini.with_structured_output() to ensure the resulting chain
    also benefits from 429 retry logic.
    """

    def __init__(self, runnable) -> None:
        # Store without triggering Pydantic field validation
        object.__setattr__(self, "_runnable", runnable)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_runnable"), name)

    # Required by RunnableSerializable
    def get_input_schema(self, config=None):
        r = object.__getattribute__(self, "_runnable")
        return r.get_input_schema(config) if hasattr(r, "get_input_schema") else super().get_input_schema(config)

    @_retry_on_resource_exhausted
    def invoke(self, input, config=None, **kwargs):
        return object.__getattribute__(self, "_runnable").invoke(input, config, **kwargs)

    @_retry_on_resource_exhausted
    def stream(self, input, config=None, **kwargs):
        return object.__getattribute__(self, "_runnable").stream(input, config, **kwargs)


# ── Provider factory ──────────────────────────────────────────────────────────

def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a chat model instance for the given (or env-configured) provider.

    Args:
        provider: One of 'gemini', 'ollama', 'vllm', 'openai', 'groq'.
                  Defaults to the LLM_PROVIDER env var (default: 'gemini').

    Returns:
        A LangChain BaseChatModel instance.  Gemini models are wrapped in
        _RateLimitedGemini for automatic 429 retry.

    Raises:
        ValueError: If required API keys are missing or the provider is unknown.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()

    # ── Gemini (Frontier API) ──────────────────────────────────────────────
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=gemini requires GEMINI_API_KEY (or GOOGLE_API_KEY) to be set."
            )
        # thinking_budget=0 disables the "thinking" feature on Gemini 3.x models.
        # When thinking is enabled, Gemini attaches a thought_signature to every
        # tool call.  On subsequent agent turns the checkpointer replays the
        # previous AIMessage (including its tool calls) back to the model, but the
        # thought_signature is not preserved — Gemini then raises:
        #   "Function call is missing a thought_signature in functionCall parts"
        # Disabling thinking entirely avoids this at zero cost to answer quality
        # for the retrieval and routing tasks this model performs.
        base_model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            google_api_key=api_key,
            temperature=0,
            thinking_budget=0,
        )
        # Wrap with rate-limit retry proxy to survive free-tier 429 errors
        return _RateLimitedGemini(base_model)

    # ── vLLM (OpenAI-compatible GPU inference server) ──────────────────────
    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            temperature=0,
        )

    # ── Ollama (local model server) ────────────────────────────────────────
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    # ── OpenAI API ─────────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    # ── Groq hosted inference ──────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY to be set.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=api_key,
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_frontier_llm() -> BaseChatModel:
    """Return the Frontier LLM (Gemini, wrapped in rate-limit retry proxy)."""
    return get_llm(provider="gemini")


def get_openweight_llm() -> BaseChatModel:
    """Return the Open-Weight LLM (Qwen via Ollama or vLLM on GPU).

    Reads OPENWEIGHT_PROVIDER (default: 'ollama') to decide which backend to
    use.  If the local backend is unavailable *and* a GROQ_API_KEY is present,
    falls back to Groq-hosted inference so development is not blocked.

    Note: Ollama fails at *invocation time* (connection refused), not at import.
    The returned model is therefore wrapped so runtime connection errors also
    trigger the Groq fallback seamlessly.

    Raises:
        Exception: Re-raises the original error if no fallback is available.
    """
    openweight_provider = os.getenv("OPENWEIGHT_PROVIDER", "ollama").lower()
    try:
        model = get_llm(provider=openweight_provider)
    except Exception as exc:
        logger.warning("Failed to initialise %s LLM: %s", openweight_provider, exc)
        if os.getenv("GROQ_API_KEY"):
            logger.info("Falling back to Groq hosted model.")
            return get_llm(provider="groq")
        raise

    # If Groq is available, wrap the model so runtime connection errors
    # (e.g. Ollama not running) automatically fall back to Groq.
    if os.getenv("GROQ_API_KEY"):
        groq_fallback = get_llm(provider="groq")

        class _OllamaWithFallback:
            """Thin proxy: tries Ollama, falls back to Groq on ConnectionError."""
            def __getattr__(self, name):
                return getattr(model, name)

            def _call_with_fallback(self, method_name, *args, **kwargs):
                try:
                    return getattr(model, method_name)(*args, **kwargs)
                except Exception as exc:
                    err = str(exc).lower()
                    if "connection" in err or "refused" in err or "connect" in err:
                        logger.warning(
                            "Ollama connection failed (%s). Falling back to Groq.", exc
                        )
                        return getattr(groq_fallback, method_name)(*args, **kwargs)
                    raise

            def invoke(self, *args, **kwargs):
                return self._call_with_fallback("invoke", *args, **kwargs)

            def stream(self, *args, **kwargs):
                return self._call_with_fallback("stream", *args, **kwargs)

            def batch(self, *args, **kwargs):
                return self._call_with_fallback("batch", *args, **kwargs)

            def bind_tools(self, *args, **kwargs):
                return model.bind_tools(*args, **kwargs)

            def with_structured_output(self, *args, **kwargs):
                return model.with_structured_output(*args, **kwargs)

            def __or__(self, other):
                return model.__or__(other)

            def __ror__(self, other):
                return model.__ror__(other)

        return _OllamaWithFallback()

    return model


def get_llm_by_tier(tier: str = "gemini") -> BaseChatModel:
    """Resolve a routing-tier string to a model instance.

    Accepted tier values:
      'gemini' / 'frontier'              → Frontier Gemini model
      'qwen' / 'openweight' / 'local'    → Open-Weight model

    Any unrecognised value falls back to the Frontier model.

    Args:
        tier: A string label produced by the dual router node.

    Returns:
        The appropriate BaseChatModel instance.
    """
    clean = tier.strip().lower()
    if clean in ("gemini", "frontier"):
        return get_frontier_llm()
    if clean in ("qwen", "openweight", "open_weight", "local"):
        return get_openweight_llm()
    # Unrecognised tier — default to frontier
    logger.warning("Unknown model tier %r — defaulting to Frontier LLM.", tier)
    return get_frontier_llm()
