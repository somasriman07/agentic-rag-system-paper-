"""Factory for chat LLM clients.

Centralizes provider selection so the rest of the codebase (rag_graph,
btw_handler, app, evaluate) never imports a concrete LangChain chat class
directly. Add a new provider by adding one branch here.

Env vars:
    LLM_PROVIDER        ollama | openai | gemini | groq   (default: ollama)
    OLLAMA_MODEL         (default: qwen3:4b)
    OLLAMA_BASE_URL       (default: http://localhost:11434)
    OPENAI_MODEL          (default: gpt-5-mini)
    OPENAI_API_KEY        required if provider=openai
    GEMINI_MODEL          (default: gemini-2.5-flash)
    GEMINI_API_KEY        required if provider=gemini
    GROQ_MODEL            (default: llama-3.3-70b-versatile)
    GROQ_API_KEY          required if provider=groq
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(override=True)

SUPPORTED_PROVIDERS = ("ollama", "openai", "gemini", "groq")


def get_llm(provider: str | None = None) -> BaseChatModel:
    """Return a chat model instance for the given (or configured) provider.

    Imports for each provider are done lazily inside their branch so errors
    are easy to attribute to the right provider and unrelated SDKs never
    need to be importable just to construct a different one.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            temperature=0,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # NOTE: ChatGoogleGenerativeAI looks for GOOGLE_API_KEY by default,
        # but this project's docs/.env use GEMINI_API_KEY — pass it through
        # explicitly so the two names don't silently diverge.
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM_PROVIDER=gemini requires GEMINI_API_KEY (or GOOGLE_API_KEY) to be set."
            )
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=api_key,
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY to be set.")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
            api_key=api_key,
            temperature=0,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )