from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
import os

load_dotenv(override=True)

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    

@tool
def add(a: int, b:int):
    """This is addition function that adds two numbers"""
    return a + b

@tool
def multiply(a: int, b:int):
    """This is multiplication function that adds two numbers"""
    return a * b

@tool
def subtract(a: int, b:int):
    """This is subtract function that subtracts two numbers"""
    return a - b

tools = [add, multiply, subtract]

model = ChatOpenAI(
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    base_url=os.environ["AZURE_OPENAI_BASE_URL"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    use_responses_api=True,
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_message = SystemMessage(
        "You are my assistant, Please answer query to the best of your ability"
    )
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"] 
    last_msg = messages[-1]
    if not last_msg.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "end": END,
        "continue": "tools"
    }
)
graph.add_edge("tools","agent")
graph.set_entry_point("agent")
agent = graph.compile()

def print_stream(stream):
    printed_count = 0
    for state in stream:
        messages = state["messages"]
        for message in messages[printed_count:]:
            if isinstance(message, AIMessage):
                message.model_copy(
                    update={"content": message.text}
                ).pretty_print()
            else:
                message.pretty_print()
        printed_count = len(messages)
inputs = {"messages":[("user","Add 2 and 4, Mutliply with 1223, Subtract 809 and tell me a  joke")]} 
print_stream(agent.stream(inputs, stream_mode="values"))