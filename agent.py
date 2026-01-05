import os
import operator
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List

# LangChain / LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Importing tools 
from tools import check_eligibility, get_scheme_documents

load_dotenv()

# 2. Setup the Brain (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# 3. Bind Tools to the Brain
tools = [check_eligibility, get_scheme_documents]
llm_with_tools = llm.bind_tools(tools)

# 4. Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# 5. Define the Nodes 
def reasoner_node(state: AgentState):
    """
    The Planner & Evaluator.
    It looks at the conversation and decides:
    1. Do I need to call a tool? (Planner)
    2. Do I have the answer? (Evaluator)
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 6. Build the Graph
builder = StateGraph(AgentState)

# Add Nodes
builder.add_node("planner", reasoner_node)
builder.add_node("executor", ToolNode(tools)) # Built-in node to run our tools

# Define Flow
builder.add_edge(START, "planner") # Start -> Planner

# Conditional Edge:
# If Planner says "Call Tool" -> Go to Executor
# If Planner says "I have the answer" -> Go to END
builder.add_conditional_edges(
    "planner",
    tools_condition,
    {"tools": "executor", "__end__": END}
)

# Loop back: After Executor runs, go back to Planner to evaluate the result
builder.add_edge("executor", "planner")

# Compile the Brain
agent_app = builder.compile()