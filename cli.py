"""Run ingestion or start the local notebook-equivalent RAG chat."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.graph import RAGChat, build_rag_graph
from src.ingestion import build_vector_store, load_vector_store
from src.models import get_embeddings, get_gemini_llm, get_local_llm
from src.retrieval import create_reranker, create_retriever


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", type=Path, help="PDF to index into chroma_db")
    parser.add_argument("--provider", choices=("local", "gemini"), default="local")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    embeddings = get_embeddings()
    if args.ingest:
        store = build_vector_store(args.ingest, embeddings)
        print(f"Indexed {store._collection.count()} chunks from {args.ingest}.")
        return

    llm = get_local_llm() if args.provider == "local" else get_gemini_llm()
    store = load_vector_store(embeddings)
    retriever = create_retriever(store, llm)
    graph, checkpointer = build_rag_graph(llm, retriever, create_reranker())
    chat = RAGChat(graph, checkpointer)
    print("RAG chat is ready. Type 'exit' to quit or 'reset' to clear this conversation.")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            chat.reset()
            print("Conversation reset.")
            continue
        if question:
            print(f"Assistant: {chat.ask(question)['answer']}")


if __name__ == "__main__":
    main()
