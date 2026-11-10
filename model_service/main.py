from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class Pair(BaseModel):
    query: str
    document: str


class RerankRequest(BaseModel):
    pairs: list[Pair] = Field(min_length=1, max_length=256)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


def make_llm():
    provider = os.getenv("RAG_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("Set GOOGLE_API_KEY in the model service environment.")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            temperature=0.0,
        )
    if provider == "local":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            temperature=0.0,
        )
    raise RuntimeError("RAG_PROVIDER must be 'local' or 'gemini'.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    embedding_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name=embedding_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    app.state.reranker = CrossEncoder(reranker_name, device="cpu")
    app.state.llm = make_llm()
    yield


app = FastAPI(title="RAG Model Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embeddings")
def embeddings(payload: EmbeddingRequest, request: Request):
    vectors = request.app.state.embeddings.embed_documents(payload.texts)
    return {"embeddings": vectors}


@app.post("/rerank")
def rerank(payload: RerankRequest, request: Request):
    scores = request.app.state.reranker.predict(
        [(pair.query, pair.document) for pair in payload.pairs]
    )
    return {"scores": [float(score) for score in scores]}


def _langchain_message(message: ChatMessage):
    if message.role == "user":
        return HumanMessage(content=message.content)
    if message.role == "assistant":
        return AIMessage(content=message.content)
    return SystemMessage(content=message.content)


@app.post("/chat")
def chat(payload: ChatRequest, request: Request):
    try:
        response = request.app.state.llm.invoke(
            [_langchain_message(message) for message in payload.messages]
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The configured chat model request failed.") from exc
    content = response.content
    if not isinstance(content, str):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return {"content": content}
