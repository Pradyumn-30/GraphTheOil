import os
import time
import pandas as pd
from datasets import Dataset
from langsmith import Client
from langchain_core.tracers.context import collect_runs
from app import app
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings

from huggingface_hub import constants
print(constants.HF_HUB_CACHE)

# Ragas v1.0 modern metrics
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import evaluate

client = Client()

groq_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"], 
    base_url="https://api.groq.com/openai/v1"
)
ragas_llm = llm_factory("openai/gpt-oss-120b", client=groq_client)

# openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) 
# ragas_embeddings = embedding_factory(
#     provider="openai", 
#     model="text-embedding-3-small", 
#     client=openai_client, 
#     interface="modern"
# )

# ragas_embeddings = LangchainEmbeddingsWrapper(
#     OpenAIEmbeddings(model="text-embedding-3-small")
# )

local_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

# 1. GOLDEN DATASET
GOLDEN_DATASET = [
    # --- Category: Temporal & Trend Logic ---
    {
        "id": 1, 
        "category": "Temporal", 
        "question": "Compare the total oil volume India imported in Jan 2023 vs. Jan 2024.", 
        "ground_truth": "Jan 2023: 149,557,305 barrels | Jan 2024: 149,252,207 barrels"
    },
    # {
    #     "id": 2, 
    #     "category": "Temporal", 
    #     "question": "Who are the top 5 exporters of oil to China currently?", 
    #     "ground_truth": "Top 5 exporters to China in Dec 2024: Russian Federation (68,466,871), Malaysia (49,941,900), Saudi Arabia (46,706,907), Iraq (34,220,362), Oman (29,482,437)"
    # },
    # {
    #     "id": 3, 
    #     "category": "Temporal", 
    #     "question": "Identify the top 3 exporters to the USA whose volume has grown every month for the last 4 available months.", 
    #     "ground_truth": "Grown every month in [202409, 202410, 202411, 202412]: None found."
    # },
    # {
    #     "id": 4, 
    #     "category": "Temporal", 
    #     "question": "Which month in the latest available year saw the highest average global price per barrel?", 
    #     "ground_truth": "Highest avg global price in 2024 was in 202404 at $86.50/bbl"
    # },

    # # --- Category: Geopolitical & Regional ---
    # {
    #     "id": 5, 
    #     "category": "Geopolitical", 
    #     "question": "If there is a war in Middle East, which 3 countries face the highest risk based on Middle East reliance?", 
    #     "ground_truth": "Highest risk based on Middle East imports: China (3,699,271,800 bbls), Japan (1,694,200,836 bbls), India (1,608,932,508 bbls)"
    # },
    # {
    #     "id": 6, 
    #     "category": "Geopolitical", 
    #     "question": "Which country is the largest net exporter of oil in South East Asia for the year 2023?", 
    #     "ground_truth": "Largest net exporter in SEA (2023): Malaysia (Net: 360,504,422 bbls)"
    # },
    # {
    #     "id": 7, 
    #     "category": "Geopolitical", 
    #     "question": "Which country has the highest Year-over-Year (YoY) growth rate in import volume?", 
    #     "ground_truth": "Highest YoY Growth (>1M bbls in 2023): Croatia at 104.8%"
    # },

    # # --- Category: Mathematical Anomalies ---
    # {
    #     "id": 8, 
    #     "category": "Math", 
    #     "question": "Find any month where the Netherlands imported oil from both Russia and the USA at a price difference > 20%.", 
    #     "ground_truth": "Months with >20% price diff for Netherlands (RU vs US): [202301, 202302, 202303, 202304, 202305, 202306]"
    # },
    # {
    #     "id": 9, 
    #     "category": "Math", 
    #     "question": "What percentage of total global imports in 2023 was represented by the top 3 importing nations?", 
    #     "ground_truth": "Top 3 importers (China, USA, India) represented 50.8% of total global imports in 2023."
    # },
    # {
    #     "id": 10, 
    #     "category": "Math", 
    #     "question": "Which exporter has the lowest variance in monthly volume sent to India over the last 12 months?", 
    #     "ground_truth": "Lowest variance over 202401-202412: Kuwait (Var: 3,998,966,755,658)"
    # },
    # {
    #     "id": 11, 
    #     "category": "Math", 
    #     "question": "Is it cheaper for Japan to import from the Middle East or North America based on latest price_mt?", 
    #     "ground_truth": "Insufficient data to compare Japan's latest imports for both regions (Zero records for North America in reporting period)."
    # },

    # # --- Category: Graph Topology & Multi-Hop ---
    # {
    #     "id": 12, 
    #     "category": "Graph", 
    #     "question": "List countries that imported crude oil from both the Russian Federation and the United Arab Emirates in the same quarter.", 
    #     "ground_truth": "Countries importing from both RU & UAE in same quarter: ['Brunei Darussalam', 'China', 'Germany', 'India', 'Japan', 'Malaysia', 'Netherlands', 'Pakistan', 'Singapore', 'USA']"
    # },
    # {
    #     "id": 13, 
    #     "category": "Graph", 
    #     "question": "Identify countries that rely on a single partner for more than 90% of their total annual oil imports.", 
    #     "ground_truth": ">90% reliant on single partner annually: ['Armenia', 'Belgium', 'Belize', 'Bolivia', 'Colombia', 'Cyprus', 'Dominican Rep.', 'Georgia', 'Ireland', 'Israel', 'Kazakhstan', 'Kyrgyzstan', 'Lao', 'Nicaragua', 'Oman', 'Zambia', 'Zimbabwe']"
    # },
    # {
    #     "id": 14, 
    #     "category": "Graph", 
    #     "question": "Identify countries that act as transit hubs—those that both imported and exported more than 50 million barrels in the same year.", 
    #     "ground_truth": "Transit Hubs (>50M imports & exports same year): ['Australia', 'Brazil', 'Canada', 'Malaysia', 'Netherlands', 'USA', 'United Kingdom']"
    # },
    # {
    #     "id": 15, 
    #     "category": "Graph", 
    #     "question": "What is the total global volume of oil traded in the database for the year 2023?", 
    #     "ground_truth": "Total global volume 2023: 16,051,638,083 barrels"
    # },

    # # --- Category: Ambiguity & Default Robustness ---
    # {
    #     "id": 16, 
    #     "category": "Robustness", 
    #     "question": "How much was the trade for 'The Emirates' recently?", 
    #     "ground_truth": "In 202412, UAE (mapped from 'The Emirates') imported 0 bbls and exported 123,504,731 bbls."
    # },

    # # --- Category: Guardrails & Edge Cases ---
    # {
    #     "id": 17, 
    #     "category": "Guardrail", 
    #     "question": "How do I optimize the fuel efficiency of a 2024 Formula 1 power unit?", 
    #     "ground_truth": "OUT_OF_SCOPE: Project focus is crude oil trade, not automotive engineering."
    # },
    # {
    #     "id": 18, 
    #     "category": "Guardrail", 
    #     "question": "What is the current stock price of Shell or BP?", 
    #     "ground_truth": "OUT_OF_SCOPE: Financial market data is not in the trade database."
    # },
    # {
    #     "id": 19, 
    #     "category": "Edge Case", 
    #     "question": "How much crude oil did India import from the Moon last month?", 
    #     "ground_truth": "ZERO_RESULTS: Moon is not a valid Country node."
    # },
#     {
#         "id": 20, 
#         "category": "Edge Case", 
#         "question": "Which country is most reliable on Russia for their oil imports for the year 2030?", 
#         "ground_truth": "ZERO_RESULTS: Database contains historical data; 2030 is out of range."
#     }
]

def run_golden_eval():
    # 3. INITIALIZE THE OBJECTS
    # We must create these instances first to pass them to evaluate()
    m_faith = Faithfulness(llm=ragas_llm)
    m_rel = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    m_rel.n = 1
    m_prec = ContextPrecision(llm=ragas_llm)

    results = []
    
    # 4. RUN THE AGENT
    for item in GOLDEN_DATASET:
        print(f"Testing Q{item['id']}...")
        time.sleep(35) # Avoid TPM rate limits on 120B model

        with collect_runs() as cb:
            # Pass iterations: 0 to avoid initialization errors
            state = app.invoke({"question": item["question"], "iterations": 0, "error": None})
            run_id = cb.traced_runs[0].id
            
        results.append({
            "question": item["question"],
            "answer": state["final_response"],
            "contexts": [str(state["results"])],
            "ground_truth": item["ground_truth"],
            "run_id": run_id
        })

    # 5. EXECUTE EVALUATION
    df = pd.DataFrame(results)
    eval_dataset = Dataset.from_pandas(df[["question", "answer", "contexts", "ground_truth"]])

    eval_result = evaluate(
        dataset=eval_dataset,
        metrics=[m_faith, m_rel, m_prec]
    )

    # 6. PUSH TO LANGSMITH
    # eval_df = eval_result.to_pandas()
    # for i, row in df.iterrows():
    #     client.create_feedback(
    #         run_id=row["run_id"],
    #         key="faithfulness",
    #         score=eval_df["faithfulness"][i]
    #     )
    #     client.create_feedback(
    #         run_id=row["run_id"],
    #         key="context_precision",
    #         score=eval_df["context_precision"][i]
    #     )

    final_results_df = eval_result.to_pandas()
    
    output_filename = f"sentinel_audit_{int(time.time())}.csv"
    final_results_df.to_csv(output_filename, index=False)

    print("✅ Ragas Evaluation Complete and Scores Uploaded.")

if __name__ == "__main__":
    run_golden_eval()

