from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import os
import zipfile
import io
from datetime import datetime
from dotenv import load_dotenv
from .embeddings import EmbeddingsManager
from .database import VectorStore
from .chat_processor import ChatProcessor

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

class ChatMessage(BaseModel):
    message: str

def extract_text_from_content(content):
    """Extract text from various content formats."""
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        if "parts" in content:
            parts = content["parts"]
            if isinstance(parts, list):
                return " ".join(str(part) for part in parts if isinstance(part, (str, int, float)))
        elif "text" in content:
            return content["text"]
    return ""

def process_chatgpt_conversation(conversation: dict) -> dict:
    """Process a ChatGPT conversation and return structured data."""
    messages = []
    
    # Extract messages from the conversation
    for message in conversation.get("mapping", {}).values():
        if message.get("message") and isinstance(message["message"], dict):
            msg_content = message["message"].get("content", {})
            text = extract_text_from_content(msg_content)
            
            if text:  # Only add messages with actual content
                role = message["message"].get("author", {}).get("role", "user")
                create_time = message["message"].get("create_time", 0)
                
                # Convert Unix timestamp to ISO format
                timestamp = datetime.fromtimestamp(create_time).isoformat() if create_time else datetime.now().isoformat()
                
                messages.append({
                    "text": text,
                    "role": role,
                    "timestamp": timestamp
                })
    
    return {
        "conversation_id": conversation.get("id", "unknown"),
        "messages": messages
    }

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        response = chat_processor.process_message(message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_upload_stream(file_content: bytes, filename: str):
    """Generator function for processing uploads and yielding status updates."""
    try:
        processed_count = 0
        total_count = 0
        errors = []
        
        if filename.endswith('.zip'):
            # Process zip file
            with zipfile.ZipFile(io.BytesIO(file_content)) as zip_ref:
                # Look for conversations.json
                if "conversations.json" in zip_ref.namelist():
                    conversations_data = json.loads(zip_ref.read("conversations.json"))
                    total_count = len(conversations_data)
                    
                    # Process each conversation
                    for idx, conversation in enumerate(conversations_data):
                        try:
                            # Convert the conversation to our format
                            processed_data = process_chatgpt_conversation(conversation)
                            
                            if processed_data["messages"]:  # Only process if there are messages
                                # Process and store embeddings
                                chunks = embeddings_manager.process_chatgpt_export(processed_data)
                                
                                for i, chunk in enumerate(chunks):
                                    vector_store.add_vector(
                                        f"{processed_data['conversation_id']}_{i}",
                                        chunk["embedding"],
                                        {
                                            "text": chunk["text"],
                                            "conversation_id": chunk["conversation_id"],
                                            "role": chunk["role"],
                                            "timestamp": chunk["timestamp"]
                                        }
                                    )
                                processed_count += 1
                        except Exception as e:
                            errors.append(f"Error in conversation {idx + 1}: {str(e)}")
                            continue
                        
                        # Send progress update every 10 conversations
                        if (idx + 1) % 10 == 0:
                            yield json.dumps({
                                "status": "processing",
                                "processed": idx + 1,
                                "total": total_count,
                                "success_count": processed_count,
                                "error_count": len(errors)
                            }) + "\n"
                    
                    # Send final status
                    yield json.dumps({
                        "status": "complete",
                        "message": f"Successfully processed {processed_count} out of {total_count} conversations",
                        "errors": errors[:10] if errors else []
                    }) + "\n"
                else:
                    yield json.dumps({
                        "status": "error",
                        "message": "No conversations.json found in zip file"
                    }) + "\n"
        else:
            # Process single JSON file
            conversation_data = json.loads(file_content)
            chunks = embeddings_manager.process_chatgpt_export(conversation_data)
            
            for i, chunk in enumerate(chunks):
                vector_store.add_vector(
                    f"{conversation_data['conversation_id']}_{i}",
                    chunk["embedding"],
                    {
                        "text": chunk["text"],
                        "conversation_id": chunk["conversation_id"],
                        "role": chunk["role"],
                        "timestamp": chunk["timestamp"]
                    }
                )
            
            yield json.dumps({
                "status": "complete",
                "message": "Conversation history uploaded and processed successfully"
            }) + "\n"
            
    except Exception as e:
        yield json.dumps({
            "status": "error",
            "message": str(e)
        }) + "\n"

@app.post("/upload")
async def upload_conversation(file: UploadFile = File(...)):
    content = await file.read()
    return StreamingResponse(
        process_upload_stream(content, file.filename),
        media_type="text/event-stream"
    )
