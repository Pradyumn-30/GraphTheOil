import streamlit as st
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# Get the directory where app.py is located
current_dir = Path(__file__).parent
from dotenv import load_dotenv
from typing import List, Optional, TypedDict

# LangChain & Graph Components
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import load_prompt
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langgraph.graph import StateGraph, END
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
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
smaller_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
cypher_prompt = load_prompt(str(current_dir / "cypher-few-shot.yaml"))
qa_prompt_template = load_prompt(str(current_dir / "qa-prompt.yaml"))

# Neo4j Connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_USER")
)

# 3. NODE DEFINITIONS
SCOPE_SYSTEM_PROMPT = (
        "You are a strict binary classifier. Your ONLY job is to determine if the user's "
        "question is about crude oil trade, volumes, prices. "
        "DO NOT attempt to answer the question. DO NOT perform any calculations. "
        "Output ONLY a JSON object with 'in_scope' (boolean) and 'reason' (string)."
    )

def check_scope_node(state: AgentState):
    # MODERN FIX: Use structured output for the Gatekeeper too
    structured_smaller_llm = smaller_llm.with_structured_output(ScopeAnalysis, method="json_mode")
    
    with get_openai_callback() as cb:
        # No more manual parsing required
        analysis = structured_smaller_llm.invoke([
            SystemMessage(content=SCOPE_SYSTEM_PROMPT), 
            HumanMessage(content=state['question'])
        ])

    return {
        "in_scope": analysis.in_scope,
        "prompt_tokens": cb.prompt_tokens, 
        "completion_tokens": cb.completion_tokens, 
        "total_tokens": cb.total_tokens, 
        "iterations": 0, 
        "error": None
    }

def generate_cypher_node(state: AgentState):
    error_msg = f"\n\nPrevious attempt failed: {state['error']}. Fix the query" if state['error'] else ""
    formatted_prompt = cypher_prompt.format(
        question=state["question"],
        schema=graph.schema,
        error=error_msg)+"\n\nYou MUST respond with a JSON object containing the key 'cypher_query'."
    with get_openai_callback() as cb:
        try:
            structured_llm = llm.with_structured_output(CypherResponse, method="json_mode")
            response = structured_llm.invoke(formatted_prompt)
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

def execute_cypher_node(state: AgentState):     
    try:
        results = graph.query(state["cypher"])
        return {"results": results, "error": None}  
    except Exception as e:
        return {"error": str(e), "results": []}

def responder_node(state: AgentState):
    if not state["in_scope"]:
        return {"final_response": "I am sorry, but your request is outside the objectives of this oil trade intelligence project."}
    if not state["results"]:
        return {"final_response": "No matching oil trade data was found."}
    full_qa_prompt = qa_prompt_template.format(question=state["question"], context=str(state["results"]))
    with get_openai_callback() as cb:
        summary = llm.invoke(full_qa_prompt)
    return {"final_response": summary.content, "prompt_tokens": state.get("prompt_tokens", 0) + cb.prompt_tokens, "completion_tokens": state.get("completion_tokens", 0) + cb.completion_tokens, "total_tokens": state.get("total_tokens", 0) + cb.total_tokens}

# 4. GRAPH CONSTRUCTION
workflow = StateGraph(AgentState)
workflow.add_node("check_scope", check_scope_node)
workflow.add_node("generate_cypher", generate_cypher_node)
workflow.add_node("execute_cypher", execute_cypher_node)
workflow.add_node("responder", responder_node)

workflow.set_entry_point("check_scope")
workflow.add_conditional_edges("check_scope", lambda x: "generate_cypher" if x["in_scope"] else "responder")
workflow.add_edge("generate_cypher", "execute_cypher")
workflow.add_conditional_edges("execute_cypher", lambda x: "generate_cypher" if x["error"] and x["iterations"] < 2 else "responder")
workflow.add_edge("responder", END)
app = workflow.compile()


if __name__ == "__main__":
    st.set_page_config(page_title="Oil Trade and Risks", page_icon="🛢️", layout="wide")
    st.title("🛢️ Geopolitical Oil Trade and Risks Intelligence Platform")
    st.markdown("---")
    st.sidebar.header("Agent Telemetry")
    st.sidebar.info("System: Tiered Llama-3 (70B) & GPT-OSS (120B)")

    query = st.chat_input("Ask about global oil trade...")

    if query:
        st.chat_message("user").write(query)
        log_container = st.container()
        
        # 3. Use 'collect_runs' to grab the trace ID for background scoring
        with collect_runs() as cb:
            inputs = {"question": query}
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
                        if "final_response" in update:
                            st.chat_message("assistant").write(update["final_response"])
                last_node_time = time.time()
            
            run_id = cb.traced_runs[0].id # Captured for cloud tracking