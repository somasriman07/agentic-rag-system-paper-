import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()


def get_embedding_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=os.getenv(
                "HF_EMBEDDING_MODEL",
                "BAAI/bge-base-en-v1.5"
            )
        )

    elif provider == "openai":
        return OpenAIEmbeddings(
            model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small"
            )
        )

    raise ValueError(f"Unsupported Embedding Provider : {provider}")