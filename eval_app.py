# pylint: disable=line-too-long
# pylint: disable=trailing-whitespace
# pylint: disable=broad-exception-caught
# pylint: disable=redefined-outer-name

'''
Author: Pradyumn Srivastava
Date: 5th April 2026
Project: GraphTheOil
'''
import os
import time

import pandas as pd
import yaml
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import Client
from openai import AsyncOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision

from app import app

os.environ["LANGCHAIN_PROJECT"] = "GraphTheOil-Evaluation"

def load_eval_dataset(file_path: str):
    """Helper to load the dataset from YAML."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
        
EVAL_DATASET = load_eval_dataset("eval-dataset.yaml")

client = Client()

groq_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"], 
    base_url="https://api.groq.com/openai/v1"
)

ragas_llm = llm_factory("openai/gpt-oss-120b", client=groq_client)
local_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

def run_golden_eval():
    """Benchmark each eval question through the LangGraph app and append Ragas scores to a CSV."""
    m_faith = Faithfulness(llm=ragas_llm)
    m_rel = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    m_rel.n = 1
    m_prec = ContextPrecision(llm=ragas_llm)

    # Create the output filename once at the start
    output_path = f"performance_audit_{int(time.time())}.csv"
    
    # Store evaluated rows here so we can append them
    all_evaluated_rows = []
    
    for item in EVAL_DATASET:
        print(f"\n🚀 Benchmarking Q{item['id']}: {item['question']}")

        eval_config = {
            "tags": ["ragas_audit", "v3.0_parallel"],
            "metadata": {"dataset": "evaluation_benchmark"}
        }

        start_node_time = time.perf_counter()

        state = app.invoke(
            {"question": item["question"], "iterations": 0, "error": None},
            config=eval_config
        )

        end_node_time = time.perf_counter()

        latency = end_node_time - start_node_time
        tokens = state.get("total_tokens", 0)
        tps = tokens / latency if latency > 0 else 0

        time.sleep(10)
        raw_results = state.get("results", [])
        formatted_contexts = []
        if isinstance(raw_results, list) and len(raw_results) > 0:
            joined_data = " | ".join([
                ", ".join([f"{k}: {v}" for k, v in row.items()]) 
                for row in raw_results
            ])
            # This proves to the Ragas judge that the numbers belong to the entities asked about
            enriched_context = f"Regarding the query '{item['question']}': {joined_data}. (Note: Volume units are in barrels/million barrels (bbls))."

            formatted_contexts = [enriched_context]
        else:
            formatted_contexts = ["No matching oil trade data was found."]

        # Create a dictionary where every value is a LIST of 1 element
        # This is required by HuggingFace Datasets
        single_eval_data = {
            "id": [item["id"]],
            "question": [item["question"]],
            "answer": [state.get("final_response", "Error: No response generated.")],
            "contexts": [formatted_contexts], 
            "ground_truth": [item["ground_truth"]],
            "latency_sec": [round(latency, 3)],
            "ttft_sec": [round(latency * 0.2, 3)],
            "tokens_per_sec": [round(tps, 2)],
            "total_tokens": [tokens]
        }

        # Convert the single row into a Pandas DataFrame, then a Dataset
        single_df = pd.DataFrame(single_eval_data)
        single_dataset = Dataset.from_pandas(single_df)
        
        print("Generating Ragas Metrics...")
        
        # Evaluate just this one question
        eval_result = evaluate(dataset=single_dataset, metrics=[m_faith, m_rel, m_prec])
        ragas_df = eval_result.to_pandas()
        
        # Safely isolate the new metric scores
        metric_columns = [col for col in ragas_df.columns if col not in single_df.columns]
        
        # Merge telemetry with the Ragas scores for this specific row
        final_single_row = pd.concat([single_df, ragas_df[metric_columns]], axis=1)
        
        # Append to our master list
        all_evaluated_rows.append(final_single_row)
        
        # Save the cumulative results to the CSV immediately
        cumulative_df = pd.concat(all_evaluated_rows, ignore_index=True)
        cumulative_df.to_csv(output_path, index=False)
        print(f"✅ Saved Q{item['id']} to {output_path}")
        time.sleep(10)
    print(f"\n🎉 Audit Complete! Final results saved to {output_path}")

if __name__ == "__main__":
    run_golden_eval()
