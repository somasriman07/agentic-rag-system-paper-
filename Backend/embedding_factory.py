"""
Backend/embedding_factory.py
-----------------------------
Factory for embedding model clients.

Mirrors the pattern in llm_factory.py: a single public function,
`get_embedding_model()`, reads the EMBEDDING_PROVIDER env var and returns
a LangChain-compatible embedding object.  Nothing outside this module needs
to import a concrete embedding class directly.

Supported providers (set via EMBEDDING_PROVIDER):
  huggingface  — HuggingFaceEmbeddings, runs locally (default: BAAI/bge-base-en-v1.5)
  ollama       — OllamaEmbeddings, served by a local Ollama instance (default: nomic-embed-text)
  openai       — OpenAIEmbeddings via the OpenAI API (default: text-embedding-3-small)

Relevant env vars:
  EMBEDDING_PROVIDER        huggingface | ollama | openai  (default: ollama)
  HF_EMBEDDING_MODEL        model name for HuggingFace     (default: BAAI/bge-base-en-v1.5)
  OLLAMA_EMBEDDING_MODEL    model name for Ollama           (default: nomic-embed-text)
  OLLAMA_BASE_URL           Ollama server URL               (default: http://localhost:11434)
  OPENAI_EMBEDDING_MODEL    model name for OpenAI           (default: text-embedding-3-small)
"""

import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv(override=True)

# Supported provider identifiers
SUPPORTED_EMBEDDING_PROVIDERS = ("huggingface", "ollama", "openai")


def get_embedding_model() -> Embeddings:
    """Return a LangChain-compatible embedding model for the configured provider.

    Reads EMBEDDING_PROVIDER from the environment (default: 'ollama') and
    instantiates the corresponding embedding class with its model and URL
    settings.

    Raises:
        ValueError: If EMBEDDING_PROVIDER is set to an unsupported value.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER: {provider!r}. "
        f"Supported values: {', '.join(SUPPORTED_EMBEDDING_PROVIDERS)}"
    )
