from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage, trim_messages
from langgraph.graph import StateGraph, START,END,add_messages
from langgraph.graph.state import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated
from typing import TypedDict
import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from IPython.display import Image
from langgraph.checkpoint.memory import MemorySaver

api_key = os.getenv("GEMINI_API_KEY")
llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key
)

db_path = "state_db/messages.db" # Add Your DB Path
con = sqlite3.connect(db_path, check_same_thread=False)

memory_saver: SqliteSaver = SqliteSaver(conn=con)

class State(MessagesState):
    summary: str

# Function that handle the summary
def chat_with_model(state: State) -> State:
    summary = state.get("summary","")

    if summary:
        system_message = f"Summary of conversion earlir {summary}"

        messages = [SystemMessage(content=system_message) + state['messages']]

    else:
        messages = state['messages']
    
    response = llm.invoke(messages)

    return {"messages": response}


def summarise_messages(state: State):
    summary = state.get("summary","")

    if summary:
        summary_message = (
        f"Here is the summary of above conversion {summary}\n\n"
        "Please extend summary by adding more information from the conversation"
        )

    else:
        summary_message = f"please provide a summary of the conversation"

    messages = state['messages'] + [HumanMessage(content=summary_message)] 
    response = llm.invoke(messages)

    deleted_messages = [RemoveMessage(id=m.id) for m in state['messages']][:-2]

    return {"summary": response.content, "messages": deleted_messages}

def should_continue(state: State) -> State:

    """Return The next node that should be executed"""

    messages = state['messages']

    if len(messages) > 6:
        return "summarise_messages"

    return END


# Now Build the Graph

# memory_saver = MemorySaver()

builder: StateGraph = StateGraph(state_schema=State)

builder.add_node("chat_with_model",chat_with_model)
builder.add_node("summarise_messages",summarise_messages)

builder.add_edge(START, "chat_with_model")
builder.add_conditional_edges("chat_with_model",should_continue)
builder.add_edge("summarise_messages",END)

graph: CompiledGraph = builder.compile(checkpointer=memory_saver)
Image(graph.get_graph().draw_mermaid_png())

# Create a thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
output = graph.invoke({"messages": [input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()

input_message = HumanMessage(content="i like the 49ers!")
output = graph.invoke({"messages": [input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()

# Check This Code So It Get the Previous Information From The Database
input_message = HumanMessage(content="what's my name?")
output = graph.invoke({"messages": [input_message]}, config)
for m in output['messages'][-1:]:
    m.pretty_print()

# Run This command for get the summary
graph.get_state(config=config).values.get("summary","")



