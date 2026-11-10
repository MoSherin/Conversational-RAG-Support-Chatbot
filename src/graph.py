from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .rag import RAGState, create_nodes, route_retrieval, route_rewrite


def build_rag_graph(llm, retriever, reranker, checkpointer: InMemorySaver | None = None):
    checkpointer = checkpointer or InMemorySaver()
    rewrite_decision, rewrite, decision, retrieve, context, answer = create_nodes(llm, retriever, reranker)
    builder = StateGraph(RAGState)
    builder.add_node("rewrite_decision", rewrite_decision)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieval_decision", decision)
    builder.add_node("retrieve", retrieve)
    builder.add_node("context", context)
    builder.add_node("answer", answer)
    builder.add_edge(START, "rewrite_decision")
    builder.add_conditional_edges(
        "rewrite_decision",
        route_rewrite,
        {"rewrite": "rewrite", "retrieval_decision": "retrieval_decision"},
    )
    builder.add_edge("rewrite", "retrieval_decision")
    builder.add_conditional_edges("retrieval_decision", route_retrieval, {"retrieve": "retrieve", "answer": "answer"})
    builder.add_edge("retrieve", "context")
    builder.add_edge("context", "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer), checkpointer


class RAGChat:
    def __init__(self, graph, checkpointer, thread_id: str = "default"):
        self.graph = graph
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}

    def ask(self, question: str):
        return self.graph.invoke({"question": question}, config=self.config)

    def reset(self):
        self.checkpointer.delete_thread(self.thread_id)
