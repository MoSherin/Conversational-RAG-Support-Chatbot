"""LangChain adapters for the separate model inference service."""

from __future__ import annotations

import os

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def _service_url() -> str:
    return os.getenv("MODEL_SERVICE_URL", "http://models:8001").rstrip("/")


class RemoteEmbeddings(Embeddings):
    """Embeddings implementation that sends batches to the model service."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url or _service_url()).rstrip("/")
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/embeddings",
            json={"texts": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class RemoteReranker:
    """Small adapter matching the predict() interface used by the RAG graph."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url or _service_url()).rstrip("/")
        self.timeout = timeout

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        response = httpx.post(
            f"{self.base_url}/rerank",
            json={"pairs": [{"query": query, "document": document} for query, document in pairs]},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["scores"]


class RemoteChatModel(BaseChatModel):
    """Chat model adapter; actual provider credentials stay in the model service."""

    base_url: str = Field(default_factory=_service_url)
    request_timeout: float = 300.0

    @property
    def _llm_type(self) -> str:
        return "conversational-rag-model-service"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        serialized = []
        for message in messages:
            role = {"human": "user", "ai": "assistant", "system": "system"}.get(message.type, "user")
            content = message.content
            if not isinstance(content, str):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            serialized.append({"role": role, "content": content})

        response = httpx.post(
            f"{self.base_url.rstrip('/')}/chat",
            json={"messages": serialized},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        answer = response.json()["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    @property
    def _identifying_params(self) -> dict:
        return {"base_url": self.base_url}
