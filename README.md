# 🛢️ Geopolitical Oil Trade and Risks Intelligence Platform

An advanced GraphRAG (Graph Retrieval-Augmented Generation) application designed for low latency and high precision. This platform allows users to query complex global crude oil trade data, calculate geopolitical import and export risks, and analyze oil trade using natural language.

Powered by a tiered multi-agent architecture via LangGraph, it translates user queries into Cypher queries which run directly on a Neo4j knowledge graph, eliminating standard vector-search hallucinations.

---

## 🏗️ System Architecture

<img src="images/system_architecture.png" alt="System Architecture">

---

## 📈 Performance & Evaluation Metrics
Rigorously benchmarked against a golden evaluation dataset using RAGAS and tracked via LangSmith, the architecture yields good GraphRAG performance:
* **Speed & Latency:** Fast **2.51s p50 (Median)** and **5.16s p99 (scenarios where retry logic hits)** response times.
* **Throughput:** Sustained generation speeds of **1,150+ Tokens Per Second (TPS)**.
* **Context Precision (1.0):** Flawless, deterministic data retrieval powered by pure Cypher logic.
* **Faithfulness (0.974):** Near-perfect hallucination-free answer generation enforced by strict LLM guardrails.
* **Answer Relevancy (0.919):** High semantic intent-matching for complex, multi-hop queries.

---

## ✨ Key Features

* **⚡ Speculative Execution:** The pipeline routes user queries concurrently to an 8B Gatekeeper and a 120B Cypher Boss, drastically reducing Time-To-First-Token (TTFT) while maintaining strict scope control.
* **🧠 "Power Trifecta" LLM Stack:**
    * **Gatekeeper (Llama-3.1-8B-Instant):** Blazing-fast scope classification and guardrail enforcement.
    * **Cypher Boss (GPT-OSS-120B):** High-intelligence context extraction and Neo4j Cypher generation.
    * **Responder (GPT-OSS-20B):** Natural language synthesis of database outputs.
* **🛠️ Self-Healing Retry Loop:** Automatically catches Cypher syntax errors and feeds them back to the 120B model for real-time correction.
* **📊 Continuous Evaluation:** Automated offline benchmarking via **RAGAS** (Faithfulness, Context Precision, Answer Correctness) and online telemetry via **LangSmith**.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Agent Orchestration:** LangGraph
* **Inference Provider:** Groq
* **Database:** Neo4j
* **Observability:** LangSmith
* **Evaluation:** Ragas, HuggingFace

---