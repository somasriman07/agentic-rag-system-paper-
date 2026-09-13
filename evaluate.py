"""RAGAS evaluation harness for the Agentic RAG pipeline.

Runs the golden question set across multiple independent runs (default: 3)
through the compiled LangGraph pipeline, scores each run with RAGAS,
and calculates cross-run statistical averages (mean and standard deviation).

Usage:
    python evaluate.py
    NUM_EVAL_RUNS=3 python evaluate.py
"""

import argparse
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from openai import OpenAI

load_dotenv(override=True)

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory, LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from Backend.embedding_factory import get_embedding_model
from Backend.llm_factory import get_llm
from Backend.logging_config import get_logger
from Backend.paper_loader import load_document
from Backend.rag_graph import build_graph
from Backend.vector_store import add_paper, get_docstore

logger = get_logger("evaluate")

GOLDEN_PATH = "golden.json"
SESSIONS_PATH = "sessions.json"
RESULTS_PATH = "evaluation_results.json"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# Small pause between per-metric RAGAS calls to respect API limits
METRIC_DELAY_SECONDS = float(os.getenv("EVAL_METRIC_DELAY_SECONDS", "1"))
QUERY_DELAY_SECONDS = float(os.getenv("EVAL_QUERY_DELAY_SECONDS", "0.2"))


def load_golden_dataset(path: str = GOLDEN_PATH) -> list[dict]:
    """Load the hand-written (question, ground_truth) evaluation set."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"'{path}' not found. Create it with a list of "
            '{"question": ..., "ground_truth": ...} objects before running evaluation.'
        )
    if not data:
        raise SystemExit(f"'{path}' is empty — add at least one question to evaluate.")
    return data


def load_evaluation_session(path: str = SESSIONS_PATH) -> str:
    """Return the most recently created session ID to evaluate against."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            sessions = json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"'{path}' not found. Upload a paper via the Streamlit app first.")
    if not sessions:
        raise SystemExit(f"No sessions found in '{path}'. Upload a paper via the Streamlit app first.")

    session_id = max(sessions.values(), key=lambda s: s.get("created_at", ""))["id"]
    logger.info("Using session '%s' for evaluation", session_id)
    return session_id


def ensure_session_documents(session_id: str) -> None:
    """Ensure the evaluation session has documents loaded into its docstore."""
    docstore = get_docstore(session_id)
    existing_keys = list(docstore.yield_keys())
    if not existing_keys:
        pdf_path = "Documents/Openclaw_Research_Report.pdf"
        if os.path.exists(pdf_path):
            logger.info("Docstore for session '%s' is empty. Ingesting '%s'...", session_id, pdf_path)
            docs = load_document(pdf_path)
            add_paper(docs, session_id)
            logger.info("Successfully indexed %d document pages for session '%s'.", len(docs), session_id)
        else:
            logger.warning("No default PDF found at '%s' to populate docstore.", pdf_path)


def get_eval_judge():
    """Build the RAGAS judge LLM + embeddings."""
    preferred = os.getenv("EVAL_LLM_PROVIDER", "groq").lower()

    if preferred == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_api_key:
            raise ValueError("EVAL_LLM_PROVIDER=groq requires GROQ_API_KEY to be set.")
        eval_model = os.getenv("EVAL_GROQ_MODEL", "openai/gpt-oss-120b")
        groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
        )
        judge_model = llm_factory(model=eval_model, client=groq_client)
        judge_name = f"Groq ({eval_model})"
        logger.info("Using Groq (%s) as the RAGAS judge LLM.", eval_model)
    elif preferred == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        eval_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        openai_client = OpenAI(api_key=openai_api_key)
        judge_model = llm_factory(model=eval_model, client=openai_client)
        judge_name = f"OpenAI ({eval_model})"
        logger.info("Using OpenAI (%s) as the RAGAS judge LLM.", eval_model)
    else:
        judge_raw = get_llm(provider=preferred)
        judge_model = LangchainLLMWrapper(judge_raw)
        judge_name = type(judge_raw).__name__
        logger.info("Using %s as the RAGAS judge LLM.", judge_name)

    embeddings = get_embedding_model()
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return judge_model, ragas_embeddings, judge_name


def collect_pipeline_outputs(golden_data: list[dict], session_id: str, run_index: int = 1) -> dict[str, list]:
    """Run every golden question through the compiled graph and gather RAGAS inputs."""
    graph = build_graph()

    questions, answers, contexts_list, ground_truths, categories = [], [], [], [], []

    for idx, item in enumerate(golden_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        category = item.get("category", "general")
        logger.info("[Run %d] [%d/%d] Querying: %s", run_index, idx + 1, len(golden_data), question)

        config = {"configurable": {"thread_id": f"eval_run{run_index}_{uuid.uuid4()}"}}
        input_state = {
            "messages": [HumanMessage(content=question)],
            "session_id": session_id,
            "query": question,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        }

        try:
            state = graph.invoke(input_state, config)
            answer = state.get("answer") or ""
            contexts = [doc.page_content for doc in (state.get("retrieved_docs") or [])]
        except Exception as e:
            logger.error("Pipeline failed on question %r: %s", question, e)
            answer, contexts = "", []

        snippet = (answer[:80] + "...") if len(answer) > 80 else answer
        logger.info("  -> contexts: %d | answer: %s", len(contexts), snippet.replace("\n", " "))
        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(ground_truth)
        categories.append(category)

        if QUERY_DELAY_SECONDS > 0:
            time.sleep(QUERY_DELAY_SECONDS)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
        "category": categories,
    }


@retry(
    retry=retry_if_not_exception_type(KeyboardInterrupt),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _evaluate_metric(dataset: Dataset, metric, llm, embeddings):
    return evaluate(
        dataset=dataset,
        metrics=[metric],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=600),
    )


def run_single_pass(run_index: int, total_runs: int, golden_data: list[dict], session_id: str, ragas_llm, ragas_embeddings) -> dict:
    """Execute a single complete evaluation pass over the golden dataset."""
    logger.info("=" * 60)
    logger.info("STARTING EVALUATION RUN %d / %d (%d questions)", run_index, total_runs, len(golden_data))
    logger.info("=" * 60)

    outputs = collect_pipeline_outputs(golden_data, session_id, run_index=run_index)
    eval_dataset = Dataset.from_dict({
        "question": outputs["question"],
        "answer": outputs["answer"],
        "contexts": outputs["contexts"],
        "ground_truth": outputs["ground_truth"],
    })

    per_row_scores: list[dict] = [{} for _ in range(len(outputs["question"]))]

    for i, metric in enumerate(METRICS):
        logger.info("[Run %d/%d] Scoring Metric %d/%d: %s", run_index, total_runs, i + 1, len(METRICS), metric.name)
        try:
            result = _evaluate_metric(eval_dataset, metric, ragas_llm, ragas_embeddings)
            for row_idx, row_score in enumerate(result.scores):
                per_row_scores[row_idx].update(row_score)
        except Exception as e:
            logger.error("[Run %d] Metric '%s' failed: %s", run_index, metric.name, e)

        if i < len(METRICS) - 1:
            time.sleep(METRIC_DELAY_SECONDS)

    run_averages: dict[str, float] = {}
    for metric in METRICS:
        values = [
            row[metric.name]
            for row in per_row_scores
            if isinstance(row.get(metric.name), (int, float)) and not math.isnan(row[metric.name])
        ]
        if values:
            run_averages[metric.name] = sum(values) / len(values)

    return {
        "run_index": run_index,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "averages": run_averages,
        "per_question": [
            {
                "id": i + 1,
                "category": outputs["category"][i],
                "question": outputs["question"][i],
                "answer": outputs["answer"][i],
                "num_contexts": len(outputs["contexts"][i]),
                "scores": per_row_scores[i],
            }
            for i in range(len(outputs["question"]))
        ],
    }


def compute_statistics(run_results: list[dict]) -> dict:
    """Compute mean and sample standard deviation across multiple runs."""
    metric_names = [m.name for m in METRICS]
    stats: dict[str, dict] = {}

    for name in metric_names:
        scores = [
            r["averages"][name]
            for r in run_results
            if name in r.get("averages", {})
        ]
        if scores:
            mean_val = sum(scores) / len(scores)
            variance = sum((x - mean_val) ** 2 for x in scores) / len(scores)
            std_val = math.sqrt(variance)
            stats[name] = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "runs": [round(s, 4) for s in scores],
            }
        else:
            stats[name] = {"mean": 0.0, "std": 0.0, "runs": []}

    return stats


def run_evaluation(num_runs: int | None = None) -> None:
    num_runs = num_runs or int(os.getenv("NUM_EVAL_RUNS", "3"))

    golden_data = load_golden_dataset()
    session_id = load_evaluation_session()
    ensure_session_documents(session_id)

    ragas_llm, ragas_embeddings, judge_name = get_eval_judge()

    logger.info("Initializing multi-run evaluation: %d runs x %d questions", num_runs, len(golden_data))

    all_run_results = []
    for r in range(1, num_runs + 1):
        run_res = run_single_pass(r, num_runs, golden_data, session_id, ragas_llm, ragas_embeddings)
        all_run_results.append(run_res)
        logger.info("[Run %d Result] %s", r, " | ".join(f"{k}: {v:.3f}" for k, v in run_res["averages"].items()))
        if r < num_runs:
            time.sleep(2)

    overall_stats = compute_statistics(all_run_results)

    final_report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "judge_llm": judge_name,
        "pipeline_llm": os.getenv("LLM_PROVIDER", "groq") + " (" + os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b") + ")",
        "embedding_model": os.getenv("EMBEDDING_PROVIDER", "huggingface") + " (" + os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5") + ")",
        "num_runs": num_runs,
        "num_questions": len(golden_data),
        "multi_run_summary": overall_stats,
        "runs": all_run_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("=" * 65)
    logger.info("ALL %d EVALUATION RUNS COMPLETED SUCCESSFULLY", num_runs)
    logger.info("=" * 65)
    logger.info("%-22s %-12s %-12s %s", "Metric", "Mean", "Std Dev", "Run Values")
    logger.info("-" * 65)
    for name, data in overall_stats.items():
        runs_str = ", ".join(f"{v:.3f}" for v in data["runs"])
        logger.info("%-22s %-12.3f %-12.3f [%s]", name, data["mean"], data["std"], runs_str)
    logger.info("=" * 65)
    logger.info("Full multi-run report saved to %s", RESULTS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-pass RAGAS evaluation")
    parser.add_argument("--runs", type=int, default=None, help="Number of independent evaluation passes")
    args = parser.parse_args()

    run_evaluation(num_runs=args.runs)