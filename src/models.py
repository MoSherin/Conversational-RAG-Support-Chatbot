from __future__ import annotations

import os

import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "llama3.2"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the normalized MiniLM embeddings used in the notebook."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_local_llm(model_name: str = LOCAL_LLM_MODEL) -> ChatOllama:
    return ChatOllama(model=model_name, temperature=0.0)


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """Optional notebook alternative; requires GOOGLE_API_KEY in the environment."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY in your environment or .env file.")
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.7)
