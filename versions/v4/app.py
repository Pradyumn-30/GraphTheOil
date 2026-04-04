import streamlit as st
import os
import concurrent.futures
#from pathlib import Path
import time
import warnings
import logging
warnings.filterwarnings("ignore")

# Get the directory where app.py is located
#current_dir = Path(__file__).parent
from dotenv import load_dotenv
from typing import List, Optional, TypedDict

# LangChain & Graph Components
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import load_prompt
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langgraph.graph import StateGraph, END
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langchain_core.tracers.context import collect_runs
from pydantic import BaseModel, Field

load_dotenv()
ls_client = Client()

class ScopeAnalysis(BaseModel):
    in_scope: bool = Field(description="Is the question about crude oil trade?")
    reason: str = Field(description="Brief explanation for the scope decision.")

class CypherResponse(BaseModel):
    cypher_query: str = Field(description="The executable Neo4j Cypher query.")

class AgentState(TypedDict):
    question: str
    cypher: Optional[str]
    error: Optional[str]
    results: List
    iterations: int
    final_response: Optional[str]
    in_scope: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# Initialize Models (Tiered approach from v2.2)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
smaller_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0) #gatekeeping
responder_llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
cypher_prompt = load_prompt("cypher-few-shot.yaml")
qa_prompt_template = load_prompt("qa-prompt.yaml")

# Neo4j Connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_USER")
)

# Pre-calculate the latest year at startup to avoid dynamic Cypher overhead
try:
    latest_year_res = graph.query("MATCH (t:YearMonth) RETURN max(t.yearMonth) / 100 AS latest_year")
    LATEST_YEAR = int(latest_year_res[0]["latest_year"]) if latest_year_res else 2024
except Exception as e:
    print(f"Warning: Could not fetch latest year. Defaulting to 2024. Error: {e}")
    LATEST_YEAR = 2024

SCOPE_SYSTEM_PROMPT = (
        "You are a strict binary classifier. Your ONLY job is to determine if the user's"
        "question is about crude oil trade, volumes, prices."
        "DO NOT attempt to answer the question. DO NOT perform any calculations. "
        "Output ONLY a JSON object with 'in_scope' (boolean) and 'reason' (string)."
    )

def check_scope_node(state: AgentState, config: RunnableConfig):
    # Use structured output for the Gatekeeper too
    structured_smaller_llm = smaller_llm.with_structured_output(ScopeAnalysis, method="json_mode")
    
    with get_openai_callback() as cb:
        # No more manual parsing required
        analysis = structured_smaller_llm.invoke(
            [
            SystemMessage(content=SCOPE_SYSTEM_PROMPT), 
            HumanMessage(content=state['question'])
            ],
            config=config
            )

    return {    
        "in_scope": analysis.in_scope,
        "prompt_tokens": cb.prompt_tokens, 
        "completion_tokens": cb.completion_tokens, 
        "total_tokens": cb.total_tokens, 
        "iterations": 0, 
        "error": None
    }

def generate_cypher_node(state: AgentState, config: RunnableConfig):
    current_error = state.get('error')
    error_msg = f"\n\nPrevious attempt failed: {current_error}. Fix the query" if current_error else ""
    formatted_prompt = cypher_prompt.format(
        question=state["question"],
        schema=graph.schema,
        error=error_msg,
        latest_year=LATEST_YEAR)+"\n\nYou MUST respond with a JSON object containing the key 'cypher_query'."
    
    structured_llm = llm.with_structured_output(CypherResponse, method="json_mode")
    with get_openai_callback() as cb:
        try:
            response = structured_llm.invoke(formatted_prompt, config=config)
            return {
                    "cypher": response.cypher_query,
                    "error": None,
                    "iterations": state["iterations"] + 1,
                    "prompt_tokens": state.get("prompt_tokens", 0) + cb.prompt_tokens, 
                    "completion_tokens": state.get("completion_tokens", 0) + cb.completion_tokens,
                    "total_tokens": state.get("total_tokens", 0) + cb.total_tokens
                    }
        except Exception as e:  
            return {
                    "error": f"LLM Output Failure: {str(e)}",
                    "iterations": state["iterations"] + 1,
                    "prompt_tokens": state.get("prompt_tokens", 0) + cb.prompt_tokens,
                    "completion_tokens": state.get("completion_tokens", 0) + cb.completion_tokens,
                    "total_tokens": state.get("total_tokens", 0) + cb.total_tokens
                    }

def speculative_entry_node(state: AgentState, config: RunnableConfig):
    """
    OPTIMIZATION: Runs Gatekeeper and Cypher Generator simultaneously. s
    Short-circuits if Out of Scope to save latency.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    future_scope = executor.submit(check_scope_node, state, config)
    future_cypher = executor.submit(generate_cypher_node, state, config)
    
    # 1. Block until the fast 8B Gatekeeper is done (~400ms)
    scope_result = future_scope.result()
    
    if not scope_result.get("in_scope"):
        # SHORT-CIRCUIT: Return immediately, abandoning the heavy Cypher thread
        executor.shutdown(wait=False, cancel_futures=True)
        return scope_result
        
    # 2. If IN SCOPE, block until Cypher Generator finishes (~3 seconds)
    cypher_result = future_cypher.result()
    executor.shutdown(wait=False)
    
    # Merge outputs safely
    return {
        "in_scope": True,
        "cypher": cypher_result.get("cypher"),
        "error": cypher_result.get("error"),
        "iterations": cypher_result.get("iterations", 1),
        "total_tokens": state.get("total_tokens", 0) + scope_result.get("total_tokens", 0) + cypher_result.get("total_tokens", 0)
    }

def execute_cypher_node(state: AgentState):     
    try:
        results = graph.query(state["cypher"])
        return {"results": results, "error": None}  
    except Exception as e:
        return {"error": str(e), "results": []}

def responder_node(state: AgentState, config: RunnableConfig):
    if not state["in_scope"]:
        return {"final_response": "I am sorry, but your request is outside the objectives of this oil trade intelligence project."}
    if not state["results"]:
        return {"final_response": "No matching oil trade data was found."}
    full_qa_prompt = qa_prompt_template.format(
        question=state["question"],
        context=str(state["results"]),
        latest_year=LATEST_YEAR
        )
    with get_openai_callback() as cb:
        summary = responder_llm.invoke(full_qa_prompt, config=config)
    return {"final_response": summary.content, "prompt_tokens": state.get("prompt_tokens", 0) + cb.prompt_tokens, "completion_tokens": state.get("completion_tokens", 0) + cb.completion_tokens, "total_tokens": state.get("total_tokens", 0) + cb.total_tokens}

# GRAPH BUILDING (Refactored for Speculative Path)
workflow = StateGraph(AgentState)
workflow.add_node("speculative_entry", speculative_entry_node)
workflow.add_node("execute_cypher", execute_cypher_node)
# Add standalone cypher node exclusively for the retry-loop
workflow.add_node("generate_cypher_retry", generate_cypher_node) 
workflow.add_node("responder", responder_node)

# Flow logic
workflow.set_entry_point("speculative_entry")
workflow.add_conditional_edges("speculative_entry", lambda x: "execute_cypher" if x["in_scope"] else "responder")

# The Retry Loop
workflow.add_conditional_edges("execute_cypher", lambda x: "generate_cypher_retry" if x["error"] and x["iterations"] < 2 else "responder")
workflow.add_edge("generate_cypher_retry", "execute_cypher")

workflow.add_edge("responder", END)
app = workflow.compile()

if __name__ == "__main__":
    st.set_page_config(page_title="Oil Trade and Risks", page_icon="🛢️", layout="wide")
    st.title("🛢️ Geopolitical Oil Trade and Risks Intelligence Platform")
    st.markdown("---")
    st.sidebar.header("Note")
    st.sidebar.info(f"Data is updated upto {LATEST_YEAR}")
    # 1. INITIALIZE SESSION STATE
    # This creates a list to hold the chat history for the current browser session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. RENDER EXISTING HISTORY
    # Every time the app reruns, redraw the previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask about global oil trade...")

    if query:
        st.chat_message("user").write(query)

        status_placeholder = st.empty()
        status_placeholder.info("⏳ Analyzing your request...")

        log_container = st.container()
        
        # 3. Use 'collect_runs' to grab the trace ID for background scoring
        with collect_runs() as cb:
            inputs = {"question": query, "error": None, "iterations": 0}
            # Inject metadata for LangSmith cloud filtering
            config = {
                "metadata": {
                    "environment": "production",
                    "app_version": "2.3",
                    "user_tier": "standard" 
                },
                "tags": ["streamlit_ui"]
            }
            
            last_node_time = time.time()
            for output in app.stream(inputs, config=config):
                duration = time.time() - last_node_time
                for node_name, update in output.items():
                    with log_container:
                        st.sidebar.write(f"⏱️ {node_name}: {duration:.2f}s")
                    if node_name == "generate_cypher_retry":
                        status_placeholder.warning("🔄 Thinking longer for a better answer...")
                    if "final_response" in update:
                        status_placeholder.empty()
                        st.chat_message("assistant").write(update["final_response"])
                last_node_time = time.time()

            run_id = cb.traced_runs[0].id # Captured for cloud tracking