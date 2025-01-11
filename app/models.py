from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ProcessingStats:
    def __init__(self):
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.total_conversations = 0
        self.total_messages = 0
        self.current_conversation = 0
    
    def update(self, processed: int = 0, skipped: int = 0, errors: int = 0):
        self.processed += processed
        self.skipped += skipped
        self.errors += errors
        
    def start_conversation(self, total_conversations: int):
        self.total_conversations = total_conversations
        self.current_conversation += 1
        
    def to_dict(self) -> Dict[str, Any]:
        total = self.processed + self.skipped + self.errors
        progress = self.current_conversation / self.total_conversations if self.total_conversations > 0 else 0
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": total,
            "progress": progress,
            "current": self.current_conversation,
            "total_conversations": self.total_conversations
        }

class UploadResult(BaseModel):
    status: str
    results: list
    memory_stats: Dict[str, int] 