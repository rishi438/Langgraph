import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

llm = ChatOllama(model="dengcao/Qwen3-14B:Q5_K_M", temperature=0)

class StateAgent(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def process_node(state: StateAgent) -> StateAgent:
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print("\nAI: ", response.content)
    print(f"CURRENT STATE: {state['messages']}")
    return state

graph = StateGraph(StateAgent)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input!="exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    # print("AI:", conversation_history[-1].content)
    user_input = input("Enter: ")

with open("logging.txt", "w",encoding="utf-8") as file:
    file.write("Your Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content} \n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")
print("Conversation saved to logging.txt")