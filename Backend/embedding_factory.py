import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv(override=True)


def get_embedding_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HF_EMBEDDING_MODEL",
                "BAAI/bge-base-en-v1.5"
            )
        )

    elif provider == "ollama":
        return OllamaEmbeddings(
            model=os.getenv(
                "OLLAMA_EMBEDDING_MODEL",
                "nomic-embed-text"
            ),
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434"
            )
        )

    elif provider == "openai":
        return OpenAIEmbeddings(
            model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small"
            )
        )

    raise ValueError(f"Unsupported Embedding Provider: {provider}")