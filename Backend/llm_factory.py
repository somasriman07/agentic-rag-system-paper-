"""
Backend/llm_factory.py
-----------------------
Factory for chat LLM clients.

Single provider in production: Groq (openai/gpt-oss-20b).
Ollama and vLLM are retained as optional local development backends.
Gemini has been removed — all tasks (routing, retrieval, generation) now
run on Groq, which is fast, free, and has no daily hard cap.

Supported providers (set via LLM_PROVIDER):
  groq    — Groq hosted inference (default, recommended)
  ollama  — Local Ollama server
  vllm    — OpenAI-compatible vLLM inference server
  openai  — OpenAI API

Relevant env vars:
  LLM_PROVIDER      groq | ollama | vllm | openai   (default: groq)
  GROQ_MODEL        Groq model name                  (default: openai/gpt-oss-20b)
  GROQ_API_KEY      Required for Groq
  OLLAMA_MODEL      Ollama model tag                 (default: qwen2.5:3b)
  OLLAMA_BASE_URL   Ollama server base URL           (default: http://localhost:11434)
  VLLM_BASE_URL     vLLM OpenAI-compat endpoint      (default: http://localhost:8000/v1)
  VLLM_MODEL        vLLM model identifier            (default: Qwen/Qwen2.5-7B-Instruct)
  VLLM_API_KEY      vLLM API key placeholder         (default: EMPTY)

Public API:
  get_llm(provider)   — return any supported provider's model
  get_groq_llm()      — shortcut: Groq model (primary provider)
"""

import os
import logging

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(override=True)
logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("groq", "ollama", "vllm", "openai")


# ── Provider factory ──────────────────────────────────────────────────────────

def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a chat model instance for the given (or env-configured) provider.

    Args:
        provider: One of 'groq', 'ollama', 'vllm', 'openai'.
                  Defaults to the LLM_PROVIDER env var (default: 'groq').

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If required API keys are missing or the provider is unknown.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()

    # ── Groq hosted inference (primary provider) ───────────────────────────
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

    # ── Ollama (local model server) ────────────────────────────────────────
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    # ── vLLM (OpenAI-compatible GPU inference server) ──────────────────────
    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            temperature=0,
        )

    # ── OpenAI API ─────────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. "
        f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


# ── Convenience accessor ──────────────────────────────────────────────────────

def get_groq_llm() -> BaseChatModel:
    """Return the Groq LLM (primary provider for all pipeline tasks)."""
    return get_llm(provider="groq")
