from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from datetime import datetime
import json
from .database import VectorStore
from .embeddings import EmbeddingsManager
import uuid
import numpy as np

class ProcessingStats:
    def __init__(self):
        self.processed = 0
        self.skipped = 0
        self.errors = 0
        self.total = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "processed": self.processed,
            "skipped": self.skipped,
            "errors": self.errors,
            "total": self.total,
            "progress": (self.processed + self.skipped) / self.total if self.total > 0 else 0
        }

    def update(self, processed: int = 0, skipped: int = 0, errors: int = 0, total: int = 0):
        self.processed += processed
        self.skipped += skipped
        self.errors += errors
        self.total += total

class ChatProcessor:
    def __init__(self, api_key: str, vector_store: VectorStore, embeddings_manager: EmbeddingsManager):
        self.client = OpenAI(api_key=api_key)
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.conversation_contexts = {}  # Cache for conversation contexts
        
    def _update_conversation_context(self, conversation_id: str, memories: List[Dict[str, Any]]) -> None:
        """Update the cached context for a conversation."""
        if conversation_id not in self.conversation_contexts:
            self.conversation_contexts[conversation_id] = set()
            
        # Add new memory IDs to the context
        for memory in memories:
            memory_id = f"{memory['conversation_id']}:{memory['sequence']}"
            self.conversation_contexts[conversation_id].add(memory_id)
            
    def _get_conversation_memories(self, conversation_id: str, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """Get memories from both cached context and new similarity search."""
        memories = []
        
        # Get memories from cached context
        if conversation_id in self.conversation_contexts:
            for memory_id in self.conversation_contexts[conversation_id]:
                conv_id, seq = memory_id.split(":")
                try:
                    memory = self.vector_store.get_memory(conv_id, int(seq))
                    if memory:
                        memories.append(memory)
                except Exception as e:
                    print(f"Error retrieving cached memory {memory_id}: {str(e)}")
        
        # Get new relevant memories
        new_memories = self.vector_store.search_similar(
            query_vector=query_embedding,
            top_k=5,
            min_similarity=0.6
        )
        
        # Combine and deduplicate memories
        seen_ids = {f"{m['conversation_id']}:{m['sequence']}" for m in memories}
        for memory in new_memories:
            memory_id = f"{memory['conversation_id']}:{memory['sequence']}"
            if memory_id not in seen_ids:
                memories.append(memory)
                seen_ids.add(memory_id)
        
        return memories

    def _chunk_conversation(self, text: str, max_tokens: int = 300) -> List[str]:
        # Simple chunking by sentences for now
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Rough token estimation (words * 1.3)
            sentence_tokens = len(sentence.split()) * 1.3
            if current_length + sentence_tokens > max_tokens and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    def process_conversation_upload(self, conversation_data: Dict[str, Any], stats: Optional[ProcessingStats] = None) -> Tuple[str, ProcessingStats]:
        """Process a conversation upload, storing messages and metadata."""
        if stats is None:
            stats = ProcessingStats()
            
        # Extract conversation ID or generate new one
        conversation_id = conversation_data.get('id', str(uuid.uuid4()))
        
        # Get messages from the conversation
        messages = conversation_data.get('messages', [])
        if not messages:
            stats.update(skipped=1)
            return conversation_id, stats
            
        # Prepare batches for processing
        chunks_to_process = []
        chunk_metadata = []
        
        # Process each message
        for message in messages:
            try:
                # Skip if message is empty or missing required fields
                if not message.get('content'):
                    stats.update(skipped=1)
                    continue
                    
                # Create chunks from the message
                chunks = self.embeddings_manager.create_chunks(message['content'])
                
                # Prepare chunks and metadata for batch processing
                for i, chunk in enumerate(chunks):
                    key = f"conv:{conversation_id}:{message.get('sequence', 0)}_{i}"
                    if self.vector_store.vector_exists(key):
                        stats.update(skipped=1)
                        continue
                        
                    chunks_to_process.append(chunk)
                    chunk_metadata.append({
                        "key": key,
                        "metadata": {
                            "conversation_id": conversation_id,
                            "timestamp": message.get('create_time'),
                            "sequence": message.get('sequence', 0),
                            "text": chunk,
                            "role": message.get('role', 'unknown'),
                            "chunk_index": i,
                            "message_index": message.get('sequence', 0),
                            "used_in_context": False
                        }
                    })
                    
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                stats.update(errors=1)
                continue
                
        # Get embeddings in batch
        if chunks_to_process:
            try:
                embeddings = self.embeddings_manager.get_embeddings_batch(chunks_to_process)
                
                # Store vectors in batch using Redis pipeline
                pipeline = self.vector_store.redis_client.pipeline()
                for embedding, meta in zip(embeddings, chunk_metadata):
                    self.vector_store.add_vector_to_pipeline(
                        pipeline=pipeline,
                        key=meta["key"],
                        vector=embedding,
                        metadata=meta["metadata"]
                    )
                pipeline.execute()
                
                # Update stats
                stats.update(processed=len(embeddings))
                
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                stats.update(errors=len(chunks_to_process))
                
        return conversation_id, stats

    def build_context_with_memories(self, current_history: List[Dict[str, Any]], memories: List[Dict[str, Any]]) -> str:
        # Sort memories by relevance and timestamp
        sorted_memories = sorted(
            memories,
            key=lambda x: (x.get('similarity', 0), x['timestamp']),
            reverse=True
        )
        
        # Build context with natural references
        context_parts = []
        
        if memories:
            context_parts.append("Based on our previous conversations:")
            for memory in sorted_memories[:3]:  # Only use top 3 most relevant memories
                timestamp = datetime.fromisoformat(memory['timestamp'])
                time_ref = self._format_time_reference(timestamp)
                
                # Format the memory text for better readability
                text = memory['text']
                if len(text) > 100:  # Truncate long memories
                    text = text[:97] + "..."
                
                context_parts.append(f"- {time_ref}: {text}")
                
                # Mark this memory as used in context
                memory['used_in_context'] = True
                self.vector_store.update_memory_metadata(memory['conversation_id'], memory['sequence'], memory)
        
        if current_history:
            context_parts.append("\nIn our current conversation:")
            # Only show the last 3 messages for immediate context
            for msg in current_history[-3:]:
                context_parts.append(f"- {msg['role']}: {msg['text']}")
        
        return "\n".join(context_parts)

    def _format_time_reference(self, timestamp: datetime) -> str:
        now = datetime.now()
        delta = now - timestamp
        
        if delta.days == 0:
            if delta.seconds < 3600:
                return "Earlier today"
            return "Today"
        elif delta.days == 1:
            return "Yesterday"
        elif delta.days < 7:
            return f"{delta.days} days ago"
        elif delta.days < 30:
            return f"{delta.days // 7} weeks ago"
        else:
            return timestamp.strftime("%B %d")

    def process_message(self, conversation_id: str, message: str, return_stats: bool = True) -> Tuple[str, Dict[str, Any]]:
        try:
            # Get conversation history
            history = self.vector_store.get_conversation_history(conversation_id) if conversation_id else []
            
            # Create embedding for the message
            query_embedding = self.embeddings_manager.get_embedding(message)
            
            # Check if user is asking about a specific conversation or topic
            is_conversation_query = any(phrase in message.lower() for phrase in [
                "show me that conversation",
                "show me the conversation",
                "what conversation",
                "what was that",
                "pull the context",
                "show me",
                "tell me about",
                "what did we discuss"
            ])
            
            # Get memories with improved search for conversation queries
            try:
                if is_conversation_query:
                    # Increase search scope for conversation queries
                    memories = self.vector_store.search_similar(
                        query_vector=query_embedding,
                        top_k=10,  # Increased from 5 to get more context
                        min_similarity=0.5  # Lowered threshold for broader search
                    )
                else:
                    memories = self._get_conversation_memories(conversation_id, query_embedding)
                
                # Calculate similarity scores for better sorting
                for memory in memories:
                    memory_vector = np.array(self.vector_store.get_vector(f"conv:{memory['conversation_id']}:{memory['sequence']}")[0])
                    query_vector = np.array(query_embedding)
                    memory['similarity'] = np.dot(memory_vector, query_vector) / (np.linalg.norm(memory_vector) * np.linalg.norm(query_vector))
                    
            except Exception as e:
                print(f"Error getting memories: {str(e)}")
                memories = []
            
            # Update conversation context with new memories
            if conversation_id:
                self._update_conversation_context(conversation_id, memories)
            
            # Build context with error handling
            try:
                context = self.build_context_with_memories(history, memories)
            except Exception as e:
                print(f"Error building context: {str(e)}")
                context = ""
            
            # If asking about a specific conversation, get the full context
            if is_conversation_query:
                # Find the most relevant conversation
                if memories:
                    most_relevant = max(memories, key=lambda x: x.get('similarity', 0))
                    conv_id = most_relevant.get("conversation_id")
                    
                    if conv_id:
                        # Get the full conversation history
                        full_conv = self.vector_store.get_conversation_history(conv_id)
                        if full_conv:
                            timestamp = datetime.fromisoformat(full_conv[0]['timestamp'])
                            time_ref = self._format_time_reference(timestamp)
                            
                            # Format the conversation with all messages
                            context = f"Here's the conversation from {time_ref}:\n\n" + "\n".join([
                                f"{msg['role']}: {msg['text']}"
                                for msg in sorted(full_conv, key=lambda x: x['sequence'])
                            ])
                            
                            # Update the response to focus on the specific topic being asked about
                            response = self.client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are helping to recall a specific conversation. Focus on answering the user's question about that conversation, providing relevant details and context."},
                                    {"role": "system", "content": context},
                                    {"role": "user", "content": message}
                                ]
                            )
                            
                            response_text = response.choices[0].message.content
                            
                            if return_stats:
                                stats = {
                                    "memory_count": len(self.vector_store.get_all_memories()),
                                    "context_count": len([m for m in memories if m.get("used_in_context", False)]),
                                    "context": context
                                }
                                return response_text, stats
                            
                            return response_text, {}
            
            # Generate response for normal queries
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are a helpful assistant with access to conversation history. 
                    Use this context naturally in your responses, referring to past discussions when relevant. 
                    When mentioning past conversations, be specific about what was discussed.
                    If you don't find a specific conversation the user asked about, be honest and say you couldn't find it."""},
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ]
            )
            
            response_text = response.choices[0].message.content
            
            # Store the new message pair with proper metadata
            if conversation_id:
                timestamp = datetime.now().isoformat()
                messages = [
                    {
                        "role": "user",
                        "text": message,
                        "timestamp": timestamp
                    },
                    {
                        "role": "assistant",
                        "text": response_text,
                        "timestamp": timestamp
                    }
                ]
                self.vector_store.add_to_conversation(conversation_id, messages)
            
            if return_stats:
                stats = {
                    "memory_count": len(self.vector_store.get_all_memories()),
                    "context_count": len([m for m in memories if m.get("used_in_context", False)]),
                    "context": context
                }
                return response_text, stats
            
            return response_text, {}
            
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            raise
