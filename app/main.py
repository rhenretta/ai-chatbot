from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any, AsyncGenerator
from dotenv import load_dotenv
import asyncio
from sse_starlette.sse import EventSourceResponse
from .embeddings import EmbeddingsManager
from .database import VectorStore
from .chat_processor import ChatProcessor
from .models import ChatRequest, ProcessingStats, UploadResult
import zipfile
from io import BytesIO

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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Get initial memory stats
    stats = vector_store.get_memory_stats()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "initial_stats": stats
    })

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response, stats = chat_processor.process_message(
            conversation_id=request.conversation_id,
            message=request.message,
            return_stats=True
        )
        return {
            "response": response,
            "conversation_id": request.conversation_id,
            "memory_count": stats.get("memory_count", 0),
            "context_count": stats.get("context_count", 0),
            "context": stats.get("context", "")  # Include context in response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()
        
        if file.filename.endswith('.zip'):
            # Handle ZIP file
            with zipfile.ZipFile(BytesIO(content)) as zip_ref:
                # Look for conversations.json
                if 'conversations.json' not in zip_ref.namelist():
                    raise HTTPException(status_code=400, detail="No conversations.json found in ZIP file")
                
                # Read and parse conversations.json
                with zip_ref.open('conversations.json') as f:
                    conversations = json.load(f)
        else:
            # Handle direct JSON upload
            try:
                conversations = json.loads(content)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Initialize stats for all conversations
        stats = ProcessingStats()
        
        # Process each conversation
        results = []
        if isinstance(conversations, list):
            stats.total_conversations = len(conversations)
            for conv in enumerate(conversations):
                stats.start_conversation(len(conversations))
                conversation_id, _ = chat_processor.process_conversation_upload(conv, stats)
                results.append({"conversation_id": conversation_id})
                
                # Send progress update
                progress = stats.to_dict()
                print(f"Processing progress: {progress}")  # Debug log
        else:
            stats.total_conversations = 1
            stats.start_conversation(1)
            conversation_id, _ = chat_processor.process_conversation_upload(conversations, stats)
            results.append({"conversation_id": conversation_id})
        
        # Get overall memory stats
        memory_stats = vector_store.get_memory_stats()
        
        return {
            "status": "success",
            "results": results,
            "memory_stats": memory_stats
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/upload-progress")
async def upload_progress(request: Request) -> EventSourceResponse:
    async def event_generator():
        while True:
            # Get current progress from Redis
            stats = vector_store.get_memory_stats()
            yield {
                "event": "progress",
                "data": json.dumps({
                    "memory_stats": stats
                })
            }
            await asyncio.sleep(1)  # Update every second
            
            if await request.is_disconnected():
                break
                
    return EventSourceResponse(event_generator())

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    try:
        history = vector_store.get_conversation_history(conversation_id)
        memory_count = len(vector_store.get_all_memories())
        context_count = len([m for m in history if m.get("used_in_context", False)])
        return JSONResponse({
            "conversation_id": conversation_id,
            "messages": history,
            "memory_count": memory_count,
            "context_count": context_count
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get current memory and conversation statistics."""
    try:
        stats = vector_store.get_memory_stats()
        return JSONResponse(stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
