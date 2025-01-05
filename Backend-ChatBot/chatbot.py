# Agentic With Memory saver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START,END,add_messages
from langgraph.graph.state import CompiledGraph
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image
from typing_extensions import Annotated
from typing import TypedDict
import os



api_key = os.getenv("GEMINI_API_KEY")
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)

memory =  MemorySaver()

class State(TypedDict):
    messages: Annotated[list,add_messages]

# Tools
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def add(a: int, b: int) -> int:
    """Addition a and b

    Args:
        a: first int
        b: second int
    """
    return a+b


llm_with_tool = llm.bind_tools([multiply,add])

class State(TypedDict):
    messages: Annotated[list, add_messages]


builder: StateGraph = StateGraph(state_schema=State)

def chatbot(state: State) -> State:
    return {"messages": [llm_with_tool.invoke(state['messages'])]}

builder.add_node("chatbot",chatbot)
builder.add_node("tools",ToolNode([multiply,add]))

builder.add_edge(START,"chatbot")
builder.add_conditional_edges("chatbot",tools_condition)
builder.add_edge("tools","chatbot")

graph: CompiledGraph = builder.compile(checkpointer=memory)


config =  {"configurable": {"thread_id": "1"}}

events = graph.stream({"messages": [HumanMessage(content="What is 2 * 3 ?")]}, config=config, stream_mode="values")

for event in events:
    event["messages"][-1].pretty_print()