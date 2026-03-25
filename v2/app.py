import streamlit as st
import os
from pathlib import Path

# Get the directory where app.py is located
current_dir = Path(__file__).parent
import time
import base64
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
from pydantic import BaseModel, Field

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Oil Trade and Risks", page_icon="🛢️", layout="wide")

# 2. HELPER FUNCTION & CSS FOR "VERY LIGHT" BACKGROUND
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# # REPLACE 'tanker.jpg' with your actual local filename
# local_filename = 'oil-tanker.png' 

# try:
#     bin_str = get_base64_of_bin_file(local_filename) 
#     st.markdown(
#         f"""
#         <style>
#         /* 1. Full Screen Background */
#         .stApp {{
#             background-image: url("data:image/png;base64,{bin_str}");
#             background-attachment: fixed;
#             background-size: cover;
#         }}

#         /* 2. Light Watermark Overlay */
#         .stApp::before {{
#             content: "";
#             position: absolute;
#             top: 0; left: 0; width: 100%; height: 100%;
#             background-color: rgba(255, 255, 255, 0.92); 
#             z-index: -1;
#         }}
        
#         /* 3. Glassmorphism: Transparency for Chat & Code Blocks */
#         [data-testid="stChatMessage"], [data-testid="stMarkdownContainer"], .stCodeBlock {{
#             background-color: rgba(255, 255, 255, 0.5) !important;
#             border-radius: 10px;
#             padding: 10px;
#             border: 1px solid rgba(0, 0, 0, 0.05);
#         }}

#         /* 4. Corrected Typography */
#         h1, h2, h3, h4, p, span, li {{
#             color: #1E1E1E !important;
#             font-family: 'Inter', sans-serif;
#         }}

#         /* 5. Custom Sidebar Styling */
#         [data-testid="stSidebar"] {{
#             background-color: rgba(240, 242, 246, 0.8) !important;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# except FileNotFoundError:
#     st.warning("Background image not found.")

st.title("🛢️ Geopolitical Oil Trade and Risks Intelligence Platform")
st.markdown("---")

# 2. INITIALIZATION & SCHEMAS
load_dotenv()

class ScopeAnalysis(BaseModel):
    in_scope: bool = Field(description="Is the question about crude oil trade or energy logistics?")
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
smaller_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
cypher_prompt = load_prompt(str(current_dir / "cypher-few-shot.yaml"))
qa_prompt_template = load_prompt(str(current_dir / "qa-prompt.yaml"))

# Neo4j Connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URL"),
    username=os.getenv("NEO4J_USER"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_USER")
)

# 3. NODE DEFINITIONS (Ported from v2.2)
SCOPE_SYSTEM_PROMPT = "You are the Gatekeeper for an oil trade system. If a query is about trade, imports, or exports, assume it refers to crude oil (HS 2709) unless another commodity is named. Output ONLY JSON per {format_instructions}. DO NOT answer the user."

def check_scope_node(state: AgentState):
    parser = JsonOutputParser(pydantic_object=ScopeAnalysis)
    system_msg = SCOPE_SYSTEM_PROMPT.format(format_instructions=parser.get_format_instructions())
    with get_openai_callback() as cb:
        response = smaller_llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=state['question'])])
        analysis = parser.parse(response.content)
    return {"in_scope": analysis['in_scope'], "prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens, "total_tokens": cb.total_tokens, "iterations": 0, "error": None}

def generate_cypher_node(state: AgentState):
    parser = JsonOutputParser(pydantic_object=CypherResponse)
    error_msg = f"\n\nPrevious attempt failed: {state['error']}. Fix the query." if state['error'] else ""
    formatted_prompt = cypher_prompt.format(question=state["question"], schema=graph.schema) + f"\n\nOutput JSON strictly: {parser.get_format_instructions()}" + error_msg
    with get_openai_callback() as cb:
        response = llm.invoke(formatted_prompt)
        structured_data = parser.parse(response.content.replace("```json", "").replace("```", "").strip())
    return {"cypher": structured_data['cypher_query'], "iterations": state["iterations"] + 1, "prompt_tokens": state.get("prompt_tokens", 0) + cb.prompt_tokens, "completion_tokens": state.get("completion_tokens", 0) + cb.completion_tokens, "total_tokens": state.get("total_tokens", 0) + cb.total_tokens}

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

# --- 5. STREAMLIT UI LOGIC ---
st.sidebar.header("Agent Telemetry")
st.sidebar.info("System: Tiered Llama-3 (70B) & GPT-OSS (120B)")

query = st.chat_input("Ask about global oil trade...")

if query:
    st.chat_message("user").write(query)
    log_container = st.container()
    
    inputs = {"question": query}
    last_node_time = time.time()
    
    for output in app.stream(inputs):
        duration = time.time() - last_node_time
        for node_name, update in output.items():
            with log_container:
                # Use columns for a clean, horizontal "Log Entry"
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.write(f"**{node_name.upper()}**")
                    st.caption(f"⏱️ {duration:.2f}s")
                with col2:
                    if "in_scope" in update:
                        st.info(f"Gatekeeper Analysis: {'✅ IN-SCOPE' if update['in_scope'] else '❌ OUT-OF-SCOPE'}")
                    if "cypher" in update:
                        st.code(update["cypher"], language="cypher")
                    if "results" in update:
                        st.success(f"Neo4j Data: {len(update['results'])} records retrieved.")
                    if "final_response" in update:
                        st.chat_message("assistant").write(update["final_response"])
                st.divider() # Text-based separator
        last_node_time = time.time()