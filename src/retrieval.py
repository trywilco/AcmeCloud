import numpy as np
from src.config import config
from src.embeddings import embed_texts


def cosine_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    return np.dot(doc_norms, query_norm)


def retrieve(
    query: str,
    documents: list[dict],
    doc_embeddings: np.ndarray,
    top_k: int | None = None
) -> list[dict]:
    if top_k is None:
        top_k = config.top_k
    
    query_embedding = embed_texts([query])[0]
    scores = cosine_similarity(query_embedding, doc_embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in top_indices]
