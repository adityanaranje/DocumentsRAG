from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from agents.graph import app as agent_app
# from ingestion.pipeline import IngestionPipeline # Optional: Trigger via API

app = FastAPI(title="Insurance Advisory AI Agent", version="1.0.0")

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[str]] = []

class ChatResponse(BaseModel):
    answer: str
    intent: str
    context_used: Optional[List[str]] = None

@app.get("/")
def health_check():
    return {"status": "active", "system": "Insurance Advisory Agent"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint. Routes query through the Multi-Agent Graph.
    """
    try:
        # Initial state
        initial_state = {
            "input": request.message,
            "chat_history": request.chat_history or [],
            "intent": "",
            "context": [],
            "answer": "",
            "metadata_filters": {}
        }
        
        # Invoke Graph
        result = agent_app.invoke(initial_state)
        
        return ChatResponse(
            answer=result.get("answer", "No response generated."),
            intent=result.get("intent", "unknown"),
            context_used=result.get("context", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
