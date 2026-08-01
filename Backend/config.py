import os

EMBEDDING_DIM = int(
    os.getenv("EMBEDDING_DIM", "768")
)