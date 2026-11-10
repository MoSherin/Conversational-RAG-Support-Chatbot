from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.routes import router
from src.graph import build_rag_graph
from src.ingestion import load_vector_store
from src.remote_models import RemoteChatModel, RemoteEmbeddings, RemoteReranker
from src.retrieval import create_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    embeddings = RemoteEmbeddings()
    llm = RemoteChatModel()
    store = load_vector_store(
        embeddings,
        persist_directory=os.getenv("CHROMA_PATH", "chroma_db"),
    )
    retriever = create_retriever(store, llm)
    graph, checkpointer = build_rag_graph(llm, retriever, RemoteReranker())
    app.state.graph = graph
    app.state.checkpointer = checkpointer
    yield


app = FastAPI(
    title="Conversational RAG API",
    description="API for the conversational RAG application.",
    version="1.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")
