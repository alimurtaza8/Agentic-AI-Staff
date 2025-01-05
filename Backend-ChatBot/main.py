from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledGraph
from chatbot import graph  # Import your chatbot logic

app = FastAPI()

# Define input and output models
class ChatInput(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    bot_response: str

# FastAPI endpoint for chatbot
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(input_data: ChatInput):
    try:
        # Prepare state and configuration for your chatbot
        state = {"messages": [HumanMessage(content=input_data.user_message)]}
        config = {"configurable": {"thread_id": "1"}}
        
        # Process the input through the chatbot
        events = graph.stream(state, config=config, stream_mode="values")
        
        # Extract the last bot message from events
        for event in events:
            last_message = event["messages"][-1].content
        
        return ChatResponse(bot_response=last_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
