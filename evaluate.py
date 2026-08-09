import os
import json
import uuid
import warnings
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Ignore deprecation and user warnings for clean logs
warnings.filterwarnings("ignore")

# Load env variables first
load_dotenv(override=True)

from Backend.rag_graph import build_graph
from Backend.llm_factory import get_llm
from Backend.embedding_factory import get_embedding_model
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def load_evaluation_session() -> str:
    try:
        with open("sessions.json", "r", encoding="utf-8") as f:
            sessions = json.load(f)
        if not sessions:
            raise ValueError("No active sessions found in sessions.json.")
        # Return the last active session ID
        last_session_id = list(sessions.keys())[-1]
        print(f"Using active session ID for evaluation: {last_session_id}")
        return last_session_id
    except FileNotFoundError:
        print("Error: sessions.json not found. Please upload a report using the Streamlit app first.")
        exit(1)
    except Exception as e:
        print(f"Error loading session: {e}")
        exit(1)

def run_evaluation():
    # 1. Load golden dataset
    try:
        with open("golden.json", "r", encoding="utf-8") as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        print("Error: golden.json not found. Please create one.")
        exit(1)

    # 2. Get active session ID and compiled graph
    session_id = load_evaluation_session()
    graph = build_graph()

    # 3. Gather outputs from the RAG graph
    print("\nRunning test queries against the RAG pipeline...")
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for item in golden_data:
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"\nEvaluating Query: '{q}'")
        config = {"configurable": {"thread_id": f"eval_{uuid.uuid4()}"}}
        
        # Invoke the LangGraph pipeline with a fully initialized input state
        input_state = {
            "messages": [HumanMessage(content=q)],
            "session_id": session_id,
            "query": q,
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
        state = graph.invoke(input_state, config)
        
        answer = state.get("answer") or ""
        retrieved_docs = state.get("retrieved_docs") or []
        contexts = [doc.page_content for doc in retrieved_docs]
        
        print(f"Response: {answer[:100]}...")
        print(f"Contexts retrieved: {len(contexts)}")
        
        questions.append(q)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(gt)

    # 4. Initialize Ragas models
    print("\nInitializing Ragas evaluation models...")
    
    from google import genai
    from ragas.llms import llm_factory
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Use Gemini specifically for evaluation if key is available to avoid NaN scores 
    # resulting from local Ollama models failing to output the correct structure.
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        print("Using Gemini 1.5 Flash (gemini-flash-latest) for evaluation (robust structured output, high free-tier quota)")
        client = genai.Client(api_key=gemini_api_key)
        ragas_llm = llm_factory(
            "gemini-flash-latest",
            provider="google",
            client=client,
            max_retries=10
        )
    else:
        print("Warning: GEMINI_API_KEY not found. Using local LLM for evaluation (might cause NaN scores for some metrics)")
        judge_llm = get_llm()
        ragas_llm = LangchainLLMWrapper(judge_llm)

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
    )

    # 5. Build HF Dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })

    # Include all 4 core Ragas metrics
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ]

    # 6. Evaluate
    print("\nComputing Ragas metrics...")
    
    # We evaluate metrics sequentially with a small delay if using Gemini to prevent hitting API rate limits
    if gemini_api_key:
        import time
        combined_scores = []
        # Prepopulate the list of score dicts
        for _ in range(len(questions)):
            combined_scores.append({})
            
        for i, metric in enumerate(metrics):
            print(f"Evaluating metric {i+1}/{len(metrics)}: {metric.name}...")
            try:
                res = evaluate(
                    dataset=eval_dataset,
                    metrics=[metric],
                    llm=ragas_llm,
                    embeddings=embeddings,
                    run_config=RunConfig(max_workers=1, timeout=600)
                )
                # Merge individual metric scores into the list of scores
                metric_scores = res.scores
                for idx, row_score in enumerate(metric_scores):
                    combined_scores[idx].update(row_score)
            except Exception as e:
                print(f"Error evaluating metric {metric.name}: {e}")
                
            if i < len(metrics) - 1:
                print("Sleeping for 15 seconds to prevent rate limit...")
                time.sleep(15)
        
        results_out = combined_scores
    else:
        # Fallback to direct evaluate if using local model
        for metric in metrics:
            metric.llm = ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = embeddings
        run_config = RunConfig(max_workers=1, timeout=600)
        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            run_config=run_config
        )
        print(results)
        results_out = results.scores if hasattr(results, "scores") else results

    print("\n" + "="*50)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*50)
    print(results_out)
    
    # Save results as a local JSON report
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print("\nResults saved to evaluation_results.json")

if __name__ == "__main__":
    run_evaluation()
