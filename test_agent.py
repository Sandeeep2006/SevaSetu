from agent import agent_app
from langchain_core.messages import HumanMessage, SystemMessage

# 1. The Persona (System Prompt)
# We strictly tell it to speak an Indian Language (e.g., Hindi/Hinglish/Marathi)
SYSTEM_PROMPT = """
You are a helpful government service agent named 'SevaSetu'.
Your goal is to help Indian citizens find government schemes.
You have access to a database of schemes and eligibility tools.

RULES:
1. ALWAYS answer in the user's native language (Hindi, Marathi, or Hinglish).
2. If you don't know the answer, use your tools to search.
3. Be polite and concise.
"""

# 2. The User Query
# We try a query that REQUIRES the database (RAG).
user_query = "Meri age 25 hai aur main unemployed hun. Kya koi scheme hai?"

print(f"User: {user_query}")
print("--- Agent Thinking (Planner -> Executor -> Evaluator) ---")

# 3. Run the Agent
messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=user_query)
]

# We stream the steps so you can see the "Reasoning" happening
for event in agent_app.stream({"messages": messages}):
    for key, value in event.items():
        print(f"\n[Node: {key}]") 
        # Only print the text content if it exists
        last_msg = value["messages"][-1]
        if hasattr(last_msg, "content") and last_msg.content:
            print(f"Response: {last_msg.content}")
        elif hasattr(last_msg, "tool_calls"):
             print(f"Action: Calling Tool -> {last_msg.tool_calls}")