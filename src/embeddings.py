import numpy as np
from src.llm_client import get_embeddings
from src.config import config


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = get_embeddings(texts, model=config.embedding_model)
    return np.array(embeddings)


def embed_documents(documents: list[dict]) -> tuple[list[dict], np.ndarray]:
    texts = [f"{doc['title']} {doc['content']}" for doc in documents]
    embeddings = embed_texts(texts)
    return documents, embeddings
