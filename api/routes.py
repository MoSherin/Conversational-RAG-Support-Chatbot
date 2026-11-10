"""HTTP routes for the existing conversational RAG graph."""

from fastapi import APIRouter, HTTPException, Request

from src.graph import RAGChat
from api.schemas import ChatRequest, ChatResponse, ResetRequest

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    question = payload.question.strip()
    conversation_id = payload.conversation_id.strip()
    if not question or not conversation_id:
        raise HTTPException(status_code=422, detail="question and conversation_id cannot be blank")

    rag_chat = RAGChat(
        request.app.state.graph,
        request.app.state.checkpointer,
        thread_id=conversation_id,
    )
    result = rag_chat.ask(question)
    return ChatResponse(answer=result["answer"], conversation_id=conversation_id)


@router.post("/reset")
def reset(payload: ResetRequest, request: Request):
    conversation_id = payload.conversation_id.strip()
    if not conversation_id:
        raise HTTPException(status_code=422, detail="conversation_id cannot be blank")

    request.app.state.checkpointer.delete_thread(conversation_id)
    return {"status": "reset", "conversation_id": conversation_id}
