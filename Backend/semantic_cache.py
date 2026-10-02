"""
Backend/semantic_cache.py
--------------------------
Semantic cache layer backed by Valkey (Redis-compatible) + BetterDB.

Instead of exact-match caching (same text → same hash), semantic caching
checks whether an incoming query is *semantically similar* to a previously
answered one.  If similarity ≥ SEMANTIC_CACHE_THRESHOLD (default 0.5), the
cached answer is returned immediately — no LLM call required.

Architecture
~~~~~~~~~~~~
  User query
      │
      ▼
  Embed query with BAAI/bge-base-en-v1.5  (same model as the RAG pipeline)
      │
      ▼
  Vector similarity search in Valkey
      │
  ┌───┴───────────────────┐
  │ similarity ≥ 0.5      │ similarity < 0.5
  ▼                       ▼
  Cache HIT               Cache MISS
  Return stored answer    Call LLM → store (query, answer) → return answer

Exposed API
~~~~~~~~~~~
  get_semantic_cache()       → SemanticCache singleton (lazy-init, safe to import)
  cache_check(query)         → CacheCheckResult  (hit/miss, similarity, cost_saved)
  cache_store(query, answer) → None
  cache_stats()              → dict  (hits, misses, hit_rate, cost_saved_usd)

All functions return gracefully if Valkey is unavailable — the cache is
treated as optional infrastructure, never a hard dependency.
"""

import inspect
import logging
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.5"))
VALKEY_HOST              = os.getenv("VALKEY_HOST", "localhost")
VALKEY_PORT              = int(os.getenv("VALKEY_PORT", "6379"))
CACHE_NAME               = "papeer_rag_cache"


# ── Async helper ──────────────────────────────────────────────────────────────
# We keep one persistent background event loop for all cache operations.
# This avoids the "Event loop is closed" error that occurs when asyncio.run()
# creates and destroys a loop between calls, invalidating the async Valkey
# client's internal connection pool.

import threading

_cache_loop: "asyncio.AbstractEventLoop | None" = None
_cache_loop_lock = threading.Lock()


def _get_or_create_loop() -> "asyncio.AbstractEventLoop":
    """Return the persistent background event loop, creating it if needed."""
    import asyncio
    global _cache_loop
    with _cache_loop_lock:
        if _cache_loop is None or _cache_loop.is_closed():
            _cache_loop = asyncio.new_event_loop()
            t = threading.Thread(target=_cache_loop.run_forever, daemon=True)
            t.start()
        return _cache_loop


def _run_async(coro):
    """Submit *coro* to the persistent background loop and block until done."""
    import asyncio
    loop = _get_or_create_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


# ── Embedding function adapter ────────────────────────────────────────────────
# BetterDB expects:  EmbedFn = Callable[[str], Awaitable[list[float]]]
# The RAG pipeline uses a synchronous HuggingFace embedding model.
# This adapter wraps the sync call in an async function.

def _make_embed_fn():
    """Return an async embed function backed by the RAG pipeline's embedding model.

    Reuses the same CacheBackedEmbeddings instance from vector_store so
    disk-cached embeddings are also reused here.
    """
    from Backend.vector_store import embeddings as _rag_embeddings

    async def embed_fn(text: str) -> list[float]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _rag_embeddings.embed_query, text
        )

    return embed_fn


# ── Singleton factory ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_semantic_cache():
    """Return the initialised SemanticCache singleton.

    Lazy and cached — first call connects to Valkey and initialises the index,
    subsequent calls return the same object instantly.

    Returns None if Valkey is unavailable so all callers degrade gracefully.
    """
    try:
        from valkey.asyncio import Valkey as AsyncValkey
        from betterdb_semantic_cache.semantic_cache import SemanticCache
        from betterdb_semantic_cache.types import SemanticCacheOptions

        client = AsyncValkey(host=VALKEY_HOST, port=VALKEY_PORT)
        # Verify connectivity synchronously before handing to the cache
        _run_async(client.ping())

        cache = SemanticCache(
            SemanticCacheOptions(
                client=client,
                embed_fn=_make_embed_fn(),
                name=CACHE_NAME,
                default_threshold=SEMANTIC_CACHE_THRESHOLD,
                default_ttl=None,            # no expiry
                use_default_cost_table=True, # enables cost_saved calculation
            )
        )
        init_result = cache.initialize()
        if inspect.iscoroutine(init_result):
            _run_async(init_result)
        logger.info(
            "Semantic cache initialised (Valkey %s:%s, threshold=%.2f)",
            VALKEY_HOST, VALKEY_PORT, SEMANTIC_CACHE_THRESHOLD,
        )
        return cache

    except Exception as exc:
        logger.warning("Semantic cache unavailable (%s). Continuing without it.", exc)
        return None


# ── Public helpers ────────────────────────────────────────────────────────────

def cache_check(query: str):
    """Check whether *query* has a semantically similar cached answer.

    Returns a CacheCheckResult with:
        .hit          True if a similar answer was found
        .response     The cached answer string (or None on miss)
        .similarity   Cosine similarity score 0–1 (or None on miss)
        .cost_saved   Estimated USD cost saved by this cache hit (or None)

    Returns None if the cache is unavailable or check fails.
    """
    cache = get_semantic_cache()
    if cache is None:
        return None
    try:
        from betterdb_semantic_cache.types import CacheCheckOptions
        result = cache.check(query, CacheCheckOptions())
        if inspect.iscoroutine(result):
            result = _run_async(result)
        return result
    except Exception as exc:
        logger.warning("Cache check failed (%s). Treating as miss.", exc)
        return None


def cache_store(query: str, answer: str) -> None:
    """Store the (query, answer) pair in the semantic cache.

    Silently ignores all errors — a failed store must never crash the pipeline.
    """
    cache = get_semantic_cache()
    if cache is None:
        return
    try:
        result = cache.store(query, answer)
        if inspect.iscoroutine(result):
            _run_async(result)
    except Exception as exc:
        logger.warning("Cache store failed (%s). Skipping.", exc)


def cache_stats() -> dict:
    """Return cache statistics as a plain dict for the sidebar UI panel.

    Keys: hits, misses, total, hit_rate (%), cost_saved_usd
    Returns an empty dict if the cache is unavailable.
    """
    cache = get_semantic_cache()
    if cache is None:
        return {}
    try:
        stats = cache.stats()
        if inspect.iscoroutine(stats):
            stats = _run_async(stats)
        return {
            "hits":           stats.hits,
            "misses":         stats.misses,
            "total":          stats.total,
            "hit_rate":       round(stats.hit_rate * 100, 1),
            "cost_saved_usd": round(stats.cost_saved_micros / 1_000_000, 4),
        }
    except Exception as exc:
        logger.warning("Cache stats failed (%s).", exc)
        return {}
