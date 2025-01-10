import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import json
from dotenv import load_dotenv

from .database import VectorStore
from .embeddings import EmbeddingsManager
from .chat_processor import ChatProcessor

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Memory RAG System")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize components
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

vector_store = VectorStore()
embeddings_manager = EmbeddingsManager(OPENAI_API_KEY)
chat_processor = ChatProcessor(OPENAI_API_KEY, vector_store, embeddings_manager)

class ChatMessage(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def root(request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        response = chat_processor.process_message(message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_content = content.decode()
        
        # Process the ChatGPT export file
        chunks = embeddings_manager.process_chatgpt_export(file_content)
        
        # Clear existing data
        vector_store.clear_all()
        
        # Store chunks with their embeddings
        for i, chunk in enumerate(chunks):
            # Create embedding for the chunk
            embedding = embeddings_manager.create_embedding(chunk['text'])
            
            # Store in vector database
            vector_store.add_vector(
                key=f"{chunk['conversation_id']}_{chunk['chunk_index']}",
                vector=embedding,
                metadata=chunk
            )
        
        return {"message": f"Successfully processed {len(chunks)} conversation chunks"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
