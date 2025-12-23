from src.config import config
from src.data_loader import load_documents
from src.embeddings import embed_documents
from src.retrieval import retrieve
from src.llm_client import generate_response

_documents: list[dict] | None = None
_embeddings = None


def initialize():
    global _documents, _embeddings
    _documents = load_documents()
    _documents, _embeddings = embed_documents(_documents)
    print(f"Initialized RAG pipeline with {len(_documents)} documents")


def build_context(retrieved_docs: list[dict]) -> str:
    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(f"[{doc['source'].upper()}] {doc['title']}\n{doc['content']}")
    return "\n\n---\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    return f"""You are a helpful AcmeCloud support assistant. Answer the question based on the provided context.

Context:
{context}

Question: {question}

Provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so."""


def answer_question(question: str) -> str:
    global _documents, _embeddings
    
    if _documents is None or _embeddings is None:
        initialize()
    
    retrieved_docs = retrieve(question, _documents, _embeddings)
    context = build_context(retrieved_docs)
    prompt = build_prompt(question, context)
    return generate_response(prompt, model=config.llm_model)
