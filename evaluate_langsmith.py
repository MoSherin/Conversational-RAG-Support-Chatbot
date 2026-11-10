"""
Evaluate the existing Magnify LangGraph RAG chatbot in LangSmith.

Expected project layout:
    main.py
    src/
    chroma_db/
    evaluation/magnify_rag_eval_complete.jsonl
    evaluation/evaluate_langsmith.py

Commands:
    python evaluate_langsmith.py --upload-dataset
    python evaluate_langsmith.py --limit 5
    python evaluate_langsmith.py

Set LANGSMITH_API_KEY and optionally LANGSMITH_PROJECT in your environment/.env.
Provider is controlled by MAGNIFY_PROVIDER=local or gemini.
For the judge, MAGNIFY_JUDGE_PROVIDER can be local or gemini. By default it reuses
the target LLM to avoid loading a second local model into memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import Client
from langsmith.evaluation import evaluate

from src.graph import build_rag_graph
from src.ingestion import load_vector_store
from src.models import get_embeddings, get_gemini_llm, get_local_llm
from src.retrieval import create_reranker, create_retriever


DATASET_NAME = "Magnify FAQ - Conversational RAG Evaluation"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "magnify_rag_eval_complete.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path.resolve()}\n"
            "Place magnify_rag_eval_complete.jsonl in the evaluation folder, "
            "or pass --dataset-path PATH."
        )
    items = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not all(key in item for key in ("inputs", "outputs", "metadata")):
                raise ValueError(
                    f"Line {line_number} must contain inputs, outputs, and metadata."
                )
            items.append(item)
    return items


def upload_dataset(client: Client, dataset_path: Path) -> None:
    examples = load_jsonl(dataset_path)
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset already exists: {DATASET_NAME} (id={dataset.id})")
        print("Skipping upload to avoid duplicating examples.")
        print("If you need to refresh it, create a new dataset name/version or remove the old dataset in LangSmith.")
        return
    except Exception:
        pass

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Magnify FAQ conversational RAG evaluation: standard questions, "
            "multi-turn follow-ups, missing-information cases, and robustness edge cases."
        ),
    )
    for item in examples:
        client.create_example(
            dataset_id=dataset.id,
            inputs=item["inputs"],
            outputs=item["outputs"],
            metadata=item.get("metadata", {}),
        )
    print(f"Uploaded {len(examples)} examples to '{DATASET_NAME}'.")
    print(f"Dataset ID: {dataset.id}")


def create_target():
    provider = os.getenv("MAGNIFY_PROVIDER", "local").lower()
    if provider not in {"local", "gemini"}:
        raise ValueError("MAGNIFY_PROVIDER must be 'local' or 'gemini'.")

    embeddings = get_embeddings()
    llm = get_local_llm() if provider == "local" else get_gemini_llm()
    vector_store = load_vector_store(embeddings)
    retriever = create_retriever(vector_store, llm)
    reranker = create_reranker()
    graph, _checkpointer = build_rag_graph(llm, retriever, reranker)

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", "")).strip()
        chat_history = inputs.get("chat_history") or []
        messages = []
        for turn in chat_history:
            role = str(turn.get("role", "")).lower()
            content = str(turn.get("content", ""))
            if role in {"user", "human"}:
                messages.append(HumanMessage(content=content))
            elif role in {"assistant", "ai"}:
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=content))

        # A unique thread per dataset example prevents conversation state leaking
        # from one evaluation example into another. The provided history is seeded
        # directly so follow-up questions are evaluated against the intended turns.
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}}
        state = graph.invoke(
            {"question": question, "messages": messages},
            config=config,
        )
        docs = state.get("documents") or []
        sources = []
        for doc in docs:
            sources.append({
                "page_content": getattr(doc, "page_content", str(doc)),
                "metadata": getattr(doc, "metadata", {}) or {},
            })

        return {
            "answer": state.get("answer", ""),
            "rewritten_question": state.get("rewritten_question", question),
            "should_retrieve": state.get("should_retrieve"),
            "retrieved_context": state.get("context", ""),
            "retrieved_documents": sources,
        }

    return target, llm


def _judge_llm(target_llm):
    provider = os.getenv("MAGNIFY_JUDGE_PROVIDER", "reuse").lower()
    if provider == "reuse":
        return target_llm
    if provider == "local":
        return get_local_llm()
    if provider == "gemini":
        return get_gemini_llm()
    raise ValueError("MAGNIFY_JUDGE_PROVIDER must be reuse, local, or gemini.")


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
    return {"score": 0, "reason": f"Could not parse judge response: {text[:500]}"}


def _judge(judge_llm, metric: str, rubric: str, question: str, answer: str,
           expected: str = "", context: str = "", history: Any = None,
           expected_behavior: str = "") -> dict[str, Any]:
    prompt = f"""
You are a careful evaluator of a conversational retrieval-augmented generation (RAG) chatbot.
Judge only using the information included below. Do not reward confident unsupported claims.
Return ONLY valid JSON with exactly these keys:
{{"score": 0 or 1, "reason": "brief explanation"}}

Metric: {metric}
Rubric:
{rubric}

Question:
{question}

Conversation history:
{json.dumps(history or [], ensure_ascii=False)}

Expected answer:
{expected}

Expected behavior:
{expected_behavior}

Retrieved context:
{context[:14000]}

Chatbot answer:
{answer}

Score 1 only if the answer satisfies the rubric. Otherwise score 0.
"""
    response = judge_llm.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in content)
    parsed = _parse_json(str(content))
    score = 1 if parsed.get("score") in (1, True, "1", "pass", "PASS") else 0
    return {"score": score, "reason": str(parsed.get("reason", ""))}


def _fields(run, example):
    inputs = getattr(example, "inputs", {}) or {}
    outputs = getattr(run, "outputs", {}) or {}
    refs = getattr(example, "outputs", {}) or {}
    metadata = getattr(example, "metadata", {}) or {}
    return inputs, outputs, refs, metadata


def build_evaluators(judge_llm):
    def answer_correctness(run, example):
        inputs, outputs, refs, _ = _fields(run, example)
        judged = _judge(
            judge_llm, "answer_correctness",
            "Pass if the chatbot answer correctly answers the question and agrees with the reference answer on material facts. Accept different wording. Penalize missing requested facts and incorrect values.",
            str(inputs.get("question", "")), str(outputs.get("answer", "")),
            str(refs.get("expected_answer", "")), str(outputs.get("retrieved_context", "")),
            inputs.get("chat_history"),
        )
        return {"key": "answer_correctness", "score": judged["score"], "comment": judged["reason"]}

    def groundedness(run, example):
        inputs, outputs, refs, _ = _fields(run, example)
        judged = _judge(
            judge_llm, "groundedness",
            "Pass only if the answer's factual claims are supported by the retrieved context or clearly framed as unknown. Unsupported specifics, invented prices, promises, or contact details fail.",
            str(inputs.get("question", "")), str(outputs.get("answer", "")),
            str(refs.get("expected_answer", "")), str(outputs.get("retrieved_context", "")),
            inputs.get("chat_history"),
        )
        return {"key": "groundedness", "score": judged["score"], "comment": judged["reason"]}

    def hallucination_check(run, example):
        inputs, outputs, refs, _ = _fields(run, example)
        judged = _judge(
            judge_llm, "hallucination_check",
            "Pass if the answer does not fabricate facts, prices, guarantees, locations, personal details, or capabilities. A correct abstention when information is missing passes. Any material unsupported assertion fails.",
            str(inputs.get("question", "")), str(outputs.get("answer", "")),
            str(refs.get("expected_answer", "")), str(outputs.get("retrieved_context", "")),
            inputs.get("chat_history"),
        )
        return {"key": "hallucination_check", "score": judged["score"], "comment": judged["reason"]}

    def retrieval_relevance(run, example):
        inputs, outputs, _refs, _ = _fields(run, example)
        context = str(outputs.get("retrieved_context", ""))
        if not context.strip():
            return {"key": "retrieval_relevance", "score": 0, "comment": "No retrieved context was returned."}
        judged = _judge(
            judge_llm, "retrieval_relevance",
            "Pass if the retrieved context contains relevant evidence that helps answer the question. Fail if context is empty, off-topic, or unrelated. Do not require every answer detail to appear in one chunk.",
            str(inputs.get("question", "")), str(outputs.get("answer", "")),
            context=context, history=inputs.get("chat_history"),
        )
        return {"key": "retrieval_relevance", "score": judged["score"], "comment": judged["reason"]}

    def multi_turn_context_resolution(run, example):
        inputs, outputs, refs, _ = _fields(run, example)
        history = inputs.get("chat_history") or []
        if not history:
            return {"key": "multi_turn_context_resolution", "score": 1, "comment": "Not applicable: no prior conversation history."}
        judged = _judge(
            judge_llm, "multi_turn_context_resolution",
            "Pass if the chatbot correctly interprets the latest question using conversation history, resolves references when possible, and does not confuse topics. If the reference is genuinely ambiguous, a clarification question is acceptable.",
            str(inputs.get("question", "")), str(outputs.get("answer", "")),
            str(refs.get("expected_answer", "")), str(outputs.get("retrieved_context", "")),
            history, str(getattr(example, "metadata", {}).get("expected_behavior", "")),
        )
        return {"key": "multi_turn_context_resolution", "score": judged["score"], "comment": judged["reason"]}

    def edge_case_pass_rate(run, example):
        inputs, outputs, refs, metadata = _fields(run, example)
        if metadata.get("category") != "edge_case":
            return {"key": "edge_case_pass_rate", "score": 1, "comment": "Not an edge-case example; excluded from edge-case reporting."}
        judged = _judge(
            judge_llm, "edge_case_pass_rate",
            "Pass if the answer follows the specified expected behavior, stays grounded, and handles ambiguity, missing information, false premises, adversarial requests, or constraints appropriately. A grounded clarification or abstention can pass when appropriate.",
            str(inputs.get("question", "")), str(outputs.get("answer", "")),
            str(refs.get("expected_answer", "")), str(outputs.get("retrieved_context", "")),
            inputs.get("chat_history"), str(metadata.get("expected_behavior", "")),
        )
        return {"key": "edge_case_pass_rate", "score": judged["score"], "comment": judged["reason"]}

    return [
        answer_correctness,
        groundedness,
        hallucination_check,
        retrieval_relevance,
        multi_turn_context_resolution,
        edge_case_pass_rate,
    ]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--upload-dataset", action="store_true", help="Upload local JSONL dataset to LangSmith and exit.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples (useful for a smoke test).")
    args = parser.parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("Set LANGSMITH_API_KEY in your environment or .env before running.")
    client = Client()

    if args.upload_dataset:
        upload_dataset(client, args.dataset_path)
        return

    if args.limit:
        # Create a temporary dataset containing only the first N examples to avoid accidentally
        # evaluating all records when smoke-testing. This is uploaded under a unique name.
        local_items = load_jsonl(args.dataset_path)[:args.limit]
        if not local_items:
            raise ValueError("No examples found in the local dataset.")
        temp_name = f"{DATASET_NAME} - smoke test {uuid.uuid4().hex[:8]}"
        temp_dataset = client.create_dataset(dataset_name=temp_name, description=f"First {len(local_items)} examples for a smoke test.")
        for item in local_items:
            client.create_example(
                dataset_id=temp_dataset.id,
                inputs=item["inputs"],
                outputs=item["outputs"],
                metadata=item.get("metadata", {}),
            )
        dataset_name = temp_name
        print(f"Smoke-test dataset: {dataset_name}")
    else:
        dataset_name = DATASET_NAME

    target, target_llm = create_target()
    judge_llm = _judge_llm(target_llm)
    evaluators = build_evaluators(judge_llm)

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="magnify-rag",
        description="Magnify FAQ RAG evaluation: correctness, groundedness, hallucinations, retrieval, multi-turn, edge cases.",
        metadata={"target_provider": os.getenv("MAGNIFY_PROVIDER", "local")},
        max_concurrency=1,
    )
    print("Evaluation submitted/completed. Open the LangSmith experiment to inspect scores and individual traces.")
    print(results)


if __name__ == "__main__":
    main()

