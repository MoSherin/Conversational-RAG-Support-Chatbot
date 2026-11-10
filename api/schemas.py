"""Pydantic request and response models for the chat API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    conversation_id: str = Field(default="default", min_length=1, max_length=200)


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str


class ResetRequest(BaseModel):
    conversation_id: str = Field(default="default", min_length=1, max_length=200)
