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
        self.chunks_processed = 0
        self.total_chunks = 0
        self.messages_in_current_conversation = 0
    
    def update(self, processed: int = 0, skipped: int = 0, errors: int = 0):
        self.processed += processed
        self.skipped += skipped
        self.errors += errors
        
    def start_conversation(self, total_conversations: int):
        """Start processing a new batch of conversations."""
        self.total_conversations = total_conversations
        self.current_conversation = 0
        self.chunks_processed = 0
        self.total_chunks = 0
        
    def start_processing_conversation(self, message_count: int):
        """Start processing a single conversation."""
        self.current_conversation += 1
        self.messages_in_current_conversation = message_count
        # Estimate total chunks based on message count and chunk size
        estimated_chunks = (message_count + 9) // 10  # Ceiling division by MESSAGE_CHUNK_SIZE (10)
        self.total_chunks += estimated_chunks
        
    def update_progress(self):
        """Update progress after processing a chunk."""
        self.chunks_processed += 1
        
    def to_dict(self) -> Dict[str, Any]:
        total = self.processed + self.skipped + self.errors
        conversation_progress = self.current_conversation / self.total_conversations if self.total_conversations > 0 else 0
        chunk_progress = self.chunks_processed / max(1, self.total_chunks)  # Avoid division by zero
        
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": total,
            "progress": chunk_progress,  # Use chunk progress as it's more granular
            "current_conversation": self.current_conversation,
            "total_conversations": self.total_conversations,
            "chunks_processed": self.chunks_processed,
            "total_chunks": self.total_chunks,
            "current_conversation_messages": self.messages_in_current_conversation
        }

class UploadResult(BaseModel):
    status: str
    results: list
    memory_stats: Dict[str, int] 