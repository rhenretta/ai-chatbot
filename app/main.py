from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
from .embeddings import EmbeddingsManager
from .database import VectorStore
from .chat_processor import ChatProcessor
from .models import ProcessingStats
import zipfile
from io import BytesIO
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Memory Chatbot")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize components
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

vector_store = VectorStore()
embeddings_manager = EmbeddingsManager(api_key)
chat_processor = ChatProcessor(api_key, vector_store, embeddings_manager)

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Get memory stats
    stats = await vector_store.get_memory_stats()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "initial_stats": stats
    })

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Processing chat request: {request.message}")
        result = await chat_processor.process_message(
            conversation_id=request.conversation_id,
            message=request.message
        )
        
        print(f"Chat processing result: {result}")
        
        # Get updated stats
        stats = await vector_store.get_memory_stats()
        
        return {
            "response": result.get("response", "I encountered an error processing your message."),
            "prompt": result.get("prompt", ""),
            "tools_used": result.get("tools_used", []),
            "conversation_id": request.conversation_id or result.get("conversation_id"),
            "stats": stats
        }
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "message": "Error processing chat message",
                "traceback": traceback.format_exc()
            }
        )

@app.get("/upload-progress")
async def upload_progress():
    """SSE endpoint for progress updates."""
    return EventSourceResponse(
        chat_processor.get_progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        content = await file.read()
        stats = ProcessingStats()
        
        if file.filename.endswith('.zip'):
            # Process ZIP file
            with zipfile.ZipFile(BytesIO(content)) as zip_ref:
                if 'conversations.json' not in zip_ref.namelist():
                    raise HTTPException(status_code=400, detail="No conversations.json found in ZIP file")
                
                with zip_ref.open('conversations.json') as f:
                    conversations = json.load(f)
        else:
            # Process single JSON file
            try:
                conversations = json.loads(content)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Process conversations
        results = []
        if isinstance(conversations, list):
            total = len(conversations)
            stats.start_conversation(total)
            
            # Start processing in background task
            background_tasks.add_task(chat_processor.process_conversations, conversations, stats)
            
            return {"status": "processing", "total": total}
        else:
            # Single conversation
            stats.start_conversation(1)
            conversation_id, updated_stats = await chat_processor.process_conversation_upload(conversations, stats)
            results.append({"conversation_id": conversation_id})
            
            # Get final stats
            memory_stats = vector_store.get_memory_stats()
            
            return {
                "status": "success",
                "results": results,
                "stats": stats.to_dict(),
                "memory_stats": memory_stats
            }
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        history = vector_store.get_conversation_history(conversation_id)
        stats = vector_store.get_memory_stats()
        return {
            "conversation_id": conversation_id,
            "messages": history,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory-stats")
async def get_memory_stats():
    """Get current memory and conversation statistics."""
    try:
        stats = await vector_store.get_memory_stats()
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
