from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import END, START
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Configure API keys and LangSmith tracing (per langsmith-trace skill)
if os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")

langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if langsmith_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key

langsmith_proj = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "ReAct-Agent"
os.environ["LANGSMITH_PROJECT"] = langsmith_proj
os.environ["LANGCHAIN_PROJECT"] = langsmith_proj


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Initialize Groq model
llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.7,
)
model = llm

def make_default_graph():
    graph_workflow=StateGraph(State)

     # Nested function defined inside make_default_graph
    def call_model(state):
        return {"messages":[model.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent=graph_workflow.compile()
    return agent

def make_alternative_graph():
    """Make a tool-calling agent"""

    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b

    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)

    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "agent")
    graph_workflow.add_edge(START, "agent")
    graph_workflow.add_conditional_edges("agent", should_continue)

    agent = graph_workflow.compile()
    return agent

agent = make_alternative_graph()

if __name__ == "__main__":
    test_input = {"messages": [("user", "What is 42 + 58?")]}
    print("Invoking agent with tracing enabled...")
    output = agent.invoke(test_input)
    print("\n--- Agent Execution Result ---")
    for msg in output["messages"]:
        role = getattr(msg, "type", "message")
        content = msg.content if msg.content else f"[Tool Calls: {getattr(msg, 'tool_calls', '')}]"
        print(f"[{role}]: {content}")

