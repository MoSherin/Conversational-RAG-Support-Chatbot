from __future__ import annotations

from langchain_classic.retrievers import MultiQueryRetriever
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def create_retriever(vector_store, llm):
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5},
    )
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


def create_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank_documents(question: str, documents: list, reranker: CrossEncoder, top_k: int = 5) -> list:
    pairs = [(question, document.page_content) for document in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    return [document for document, _ in ranked[:top_k]]
