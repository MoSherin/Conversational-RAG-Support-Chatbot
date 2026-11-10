from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import add_messages

from src.retrieval import rerank_documents


class RAGState(TypedDict, total=False):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    should_rewrite: bool
    rewritten_question: str
    should_retrieve: bool
    relevant_answer: bool
    documents: list
    context: str
    answer: str


RETRIEVAL_DECISION_PROMPT = ChatPromptTemplate.from_template("""
You are a strict binary routing classifier. Your task is to analyze an incoming user input and determine if external context/retrieval is required to give an accurate answer.

Classification Rules:
- YES: The query asks about specific documents, prior context ("it", "that", "this"), real-time/current events, private data, complex domain specifics, or factual details requiring external knowledge.
- NO: The query is generic conversation, a greeting.

Output Requirements:
- You MUST output ONLY the single word "YES" or "NO".
- Do NOT include reasoning, punctuation, prefixes, or extra words.

Examples:
User: Hello, how are you?
Classification: NO

User: What does the document say about the policy?
Classification: YES

User: What happened in the last step?
Classification: YES

User: Explain how backpropagation works.
Classification: YES

User: Can you summarize that?
Classification: YES

User: Thank you so much!
Classification: NO

Query: {question}
Classification:
""")

REWRITE_DECISION_PROMPT = ChatPromptTemplate.from_template("""
You are a query-rewrite routing classifier for a conversational RAG system.

Decide whether the latest user message needs the conversation history to be
understood as a standalone retrieval question.

Return exactly one token:
YES
or
NO

Return YES when the message depends on an earlier topic, for example it uses
references such as "it", "they", "that", "this", "the previous one", or
asks for more detail about something already discussed.

Important: a named topic does not make a question standalone. If a pronoun's
referent is missing from the latest message, return YES even when the message
also mentions a topic such as RAG, embeddings, or prompt engineering.

Return NO when the message is already a complete standalone question, or is a
greeting, acknowledgement, feedback, or casual conversation.

Question: What are embeddings?
Answer: NO

Question: Why are they useful?
Answer: YES

Question: How do they work with Prompt Engineering?
Answer: YES

Question: How do they relate to RAG?
Answer: YES

Question: How is it used in RAG?
Answer: YES

Question: Does RAG use embeddings?
Answer: NO

Question: Can you explain that further?
Answer: YES

Question: Hello
Answer: NO

Question: {question}
Answer:
""")

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Rewrite the user's latest question into a standalone question that can be understood without the conversation history.
Preserve meaning, resolve references, do not answer, and return ONLY the rewritten question.
If the latest message is already standalone, return it unchanged.
If the message is conversational, an acknowledgement, a greeting, feedback,
or otherwise not a retrieval question, return it exactly unchanged.
If the message is irrelevant to the conversation, return it exactly unchanged.
Do not turn it into a question. Do not add Chat History.

Examples:

<example>
Chat history: Assistant explained MultiQueryRetriever.
Latest question: thanks for the information
Standalone question: thanks for the information
</example>

<example>
Chat history: User discussed Chroma score thresholds.
Latest question: That makes sense, thank you!
Standalone question: That makes sense, thank you!
</example>

<example>
Chat history: User asked about document chunking.
Latest question: hello
Standalone question: hello
</example>

Conversation history:
{chat_history}
Latest question: {question}
Standalone question:
""")

RELEVANCY_GRADER_PROMPT = ChatPromptTemplate.from_template("""
You are a relevance grader.

Determine whether the context contains information that helps answer the
rewritten question.

Return exactly one token:
RELEVANT
or
IRRELEVANT

Answer RELEVANT when the context:
- directly answers the question
- contains facts, definitions, steps, examples, or details that help answer it
- is partially relevant and could contribute to a complete answer

Answer IRRELEVANT when the context:
- is about a different topic
- only shares unrelated words with the question
- does not provide useful information for answering the question

Do not use outside knowledge.
Do not infer information that is not present in the context.
When uncertain, return RELEVANT if the context could reasonably help answer
the question.


Question: {rewritten_question}
Context:
{context}
Answer:
""")


IRRELEVANT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Write one short, warm response to the user.
Let them know that you are here to help with support questions or anything
related to their customer experience.
Use natural wording and vary the phrasing each time. Do not reuse a fixed
sentence, quote these instructions, or ask unrelated questions. Keep the
response to exactly one sentence.
""")


RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant answering questions about a document.
Answer using ONLY the provided context. If the answer cannot be found in the context, say "I don't know based on the provided document."
Do not invent information. Cite every factual claim as [Source 1], [Source 2], etc., and cite only supporting sources.
Context:
----------------
{context}
----------------
Question: {question}
Answer:
""")


def _content(response) -> str:
    """Normalize chat-model response content to text for both local and Gemini paths."""
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content).strip()


def format_documents(documents: list) -> str:
    return "\n\n".join(
        f"""[Source {index}]
Document: {document.metadata.get('title', 'N/A')}
Author: {document.metadata.get('author', 'N/A')}
Page: {document.metadata.get('page_label', 'N/A')}
Chunk ID: {document.metadata.get('chunk_id', 'N/A')}

{document.page_content}"""
        for index, document in enumerate(documents, start=1)
    )


def create_nodes(llm, retriever, reranker):
    retrieval_decision_chain = RETRIEVAL_DECISION_PROMPT | llm
    rewrite_decision_chain = REWRITE_DECISION_PROMPT | llm
    rewrite_chain = REWRITE_PROMPT | llm
    rag_chain = RAG_PROMPT | llm
    relevancy_grader_chain = RELEVANCY_GRADER_PROMPT | llm
    irrelevant_answer_chain = IRRELEVANT_ANSWER_PROMPT | llm

    def rewrite_node(state: RAGState):
        messages = state.get("messages", [])
        if not messages:
            return {"rewritten_question": state["question"]}
        history = "\n".join(f"{message.type}: {message.content}" for message in messages)
        return {"rewritten_question": _content(rewrite_chain.invoke({"chat_history": history, "question": state["question"]}))}

    def rewrite_decision_node(state: RAGState):
        decision = _content(rewrite_decision_chain.invoke({"question": state["question"]})).upper()
        result = {"should_rewrite": decision == "YES"}
        if decision != "YES":
            result["rewritten_question"] = state["question"]

        return result

    def retrieval_decision_node(state: RAGState):
        # This node runs after rewrite_node. Routing on the rewritten question is
        # essential for follow-ups such as "Why are they useful?".
        question = state.get("rewritten_question", state["question"])
        decision = _content(retrieval_decision_chain.invoke({"question": question})).upper()
        return {"should_retrieve": decision == "YES"}

    def retrieval_node(state: RAGState):
        question = state["rewritten_question"]
        documents = retriever.invoke(question)
        return {"documents": rerank_documents(question, documents, reranker, top_k=5)}

    def context_node(state: RAGState):
        return {"context": format_documents(state["documents"])}

    def answer_node(state: RAGState):
        if state["should_retrieve"]:
            decision = _content(relevancy_grader_chain.invoke({"rewritten_question": state["rewritten_question"], "context": state["context"]})).upper()
            state["relevant_answer"] = decision == "RELEVANT"
            if state["relevant_answer"]:
                response = rag_chain.invoke({"context": state["context"], "question": state["rewritten_question"]})
            else:
                response = irrelevant_answer_chain.invoke({})
        else:
            response = llm.invoke([*state.get("messages", []), HumanMessage(content=state["question"])])
        answer = _content(response)
        return {"answer": answer, "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)]}

    return rewrite_decision_node, rewrite_node, retrieval_decision_node, retrieval_node, context_node, answer_node


def route_retrieval(state: RAGState) -> str:
    return "retrieve" if state["should_retrieve"] else "answer"


def route_rewrite(state: RAGState) -> str:
    return "rewrite" if state["should_rewrite"] else "retrieval_decision"
