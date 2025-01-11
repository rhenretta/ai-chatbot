import redis
import redis.asyncio as aioredis
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """Initialize Redis connections with proper configuration."""
        # Configure Redis client with connection pooling
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False,  # Changed to False to handle binary data properly
            socket_keepalive=True,  # Keep connection alive
            health_check_interval=30  # Check connection health every 30 seconds
        )
        
        # Configure async Redis client
        self.redis_async_client = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=False,  # Changed to False to handle binary data properly
            health_check_interval=30
        )
        
        # Test connection
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            print(f"Error connecting to Redis: {str(e)}")
            raise
        
    def add_vector(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Synchronous version of add_vector."""
        try:
            # Convert vector to bytes for storage
            vector_bytes = np.array(vector, dtype=np.float32).tobytes()
            
            # Store the vector and metadata
            pipeline = self.redis_client.pipeline()
            pipeline.hset(f"vector:{key}", mapping={
                "embedding": vector_bytes,
                "metadata": json.dumps(metadata)
            })
            pipeline.execute()
        except Exception as e:
            print(f"Error adding vector {key}: {str(e)}")

    async def add_vector_async(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Asynchronous version of add_vector."""
        try:
            # Convert vector to bytes for storage
            vector_bytes = np.array(vector, dtype=np.float32).tobytes()
            
            # Create pipeline and add commands without awaiting individual operations
            pipeline = self.redis_async_client.pipeline()
            pipeline.hset(f"vector:{key}", mapping={
                "embedding": vector_bytes,
                "metadata": json.dumps(metadata)
            })
            # Execute all commands in pipeline at once
            await pipeline.execute()
        except Exception as e:
            print(f"Error adding vector {key}: {str(e)}")

    async def add_vectors_batch_async(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Add multiple vectors in a single batch operation asynchronously."""
        try:
            pipeline = self.redis_async_client.pipeline()
            
            # Add all commands to pipeline without awaiting individual operations
            for key, vector, metadata in vectors:
                vector_bytes = np.array(vector, dtype=np.float32).tobytes()
                pipeline.hset(f"vector:{key}", mapping={
                    "embedding": vector_bytes,
                    "metadata": json.dumps(metadata)
                })
            
            # Execute all commands in pipeline at once
            await pipeline.execute()
        except Exception as e:
            print(f"Error in batch vector addition: {str(e)}")

    def update_memory_metadata(self, conversation_id: str, sequence: int, metadata: Dict[str, Any]) -> None:
        key = f"vector:chat:{conversation_id}_{sequence}"
        result = self.redis_client.hget(f"vector:{key}", "metadata")
        if result:
            current_metadata = json.loads(result.decode())
            current_metadata.update(metadata)
            self.redis_client.hset(f"vector:{key}", "metadata", json.dumps(current_metadata).encode())

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get the full conversation history with proper metadata and sorting."""
        # Get all keys for this conversation
        pattern = f"vector:chat:{conversation_id}_*"
        keys = self.redis_client.keys(pattern)
        
        if not keys:
            print(f"No messages found for conversation: {conversation_id}")
            return []
        
        # Get all chunks with proper error handling
        chunks = []
        for key in keys:
            try:
                result = self.redis_client.hgetall(key)
                if result and b"metadata" in result:
                    metadata = json.loads(result[b"metadata"].decode())
                    # Add default values for missing required fields
                    metadata.setdefault("sequence", 0)
                    metadata.setdefault("text", "")
                    metadata.setdefault("role", "unknown")
                    metadata.setdefault("timestamp", datetime.now().isoformat())
                    chunks.append(metadata)
            except Exception as e:
                print(f"Error retrieving message {key}: {str(e)}")
                continue
        
        # Sort by sequence number and timestamp
        sorted_chunks = sorted(
            chunks,
            key=lambda x: (x.get("sequence", 0), x.get("timestamp", ""))
        )
        
        # Group chunks by message if needed
        messages = []
        current_msg = None
        
        for chunk in sorted_chunks:
            if not current_msg or current_msg["sequence"] != chunk["sequence"]:
                if current_msg:
                    messages.append(current_msg)
                current_msg = chunk.copy()  # Create a copy to avoid modifying the original
            else:
                # Combine text if this is a continuation
                current_msg["text"] += " " + chunk["text"]
        
        if current_msg:
            messages.append(current_msg)
        
        return messages

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all stored memories/chunks across all conversations."""
        keys = self.redis_client.keys("vector:chat:*")
        memories = []
        for key in keys:
            result = self.redis_client.hget(key, "metadata")
            if result:
                memories.append(json.loads(result.decode()))
        return memories

    def search_similar(self, query_vector: List[float], top_k: int = 5, conversation_id: Optional[str] = None, min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar vectors with improved error handling and accuracy."""
        try:
            # Convert query vector to numpy for calculations
            query_np = np.array(query_vector, dtype=np.float32)
            
            # Get all vectors
            pattern = f"vector:chat:{conversation_id + '_*' if conversation_id else '*'}"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                print("No vectors found matching pattern:", pattern)
                return []
            
            results = []
            for key in keys:
                try:
                    result = self.redis_client.hgetall(key)
                    if not result or b"embedding" not in result or b"metadata" not in result:
                        continue
                        
                    vector = np.frombuffer(result[b"embedding"], dtype=np.float32)
                    metadata = json.loads(result[b"metadata"].decode())
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_np, vector) / (np.linalg.norm(query_np) * np.linalg.norm(vector))
                    
                    # Only include results above similarity threshold
                    if similarity >= min_similarity:
                        metadata['similarity'] = float(similarity)  # Convert to float for JSON serialization
                        results.append(metadata)
                        
                except Exception as e:
                    print(f"Error processing vector {key}: {str(e)}")
                    continue
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in search_similar: {str(e)}")
            return []

    def get_vector(self, key: str) -> tuple[List[float], Dict[str, Any]]:
        """Get a vector and its metadata by key."""
        try:
            result = self.redis_client.hgetall(f"vector:{key}")
            if not result:
                raise KeyError(f"No vector found for key: {key}")
                
            # Convert bytes back to vector
            vector = np.frombuffer(result[b"embedding"], dtype=np.float32).tolist()
            metadata = json.loads(result[b"metadata"].decode())
            return vector, metadata
        except Exception as e:
            print(f"Error getting vector {key}: {str(e)}")
            raise
        
    def vector_exists(self, key: str) -> bool:
        """Check if a vector already exists in Redis."""
        return self.redis_client.exists(f"vector:{key}")
    
    def get_existing_conversation_ids(self) -> set:
        """Get all unique conversation IDs from Redis."""
        keys = self.redis_client.keys("vector:chat:*")
        conversation_ids = set()
        for key in keys:
            try:
                metadata = json.loads(self.redis_client.hget(key, "metadata").decode())
                conversation_ids.add(metadata.get("conversation_id"))
            except:
                continue
        return conversation_ids
    
    async def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about stored memories."""
        try:
            # Get all keys that start with 'vector:chat:'
            keys = await self.redis_async_client.keys("vector:chat:*")
            total_memories = len(keys)
            
            # Count used memories (those that have been accessed)
            used_memories = 0
            conversation_ids = set()
            
            for key in keys:
                try:
                    metadata_bytes = await self.redis_async_client.hget(key, "metadata")
                    if metadata_bytes:
                        metadata = json.loads(metadata_bytes.decode())
                        if metadata.get("last_used"):
                            used_memories += 1
                        if "conversation_id" in metadata:
                            conversation_ids.add(metadata["conversation_id"])
                except Exception as e:
                    print(f"Error processing memory stats for key {key}: {str(e)}")
                    continue
            
            return {
                "total_memories": total_memories,
                "used_memories": used_memories,
                "total_conversations": len(conversation_ids)
            }
        except Exception as e:
            print(f"Error getting memory stats: {str(e)}")
            return {
                "total_memories": 0,
                "used_memories": 0,
                "total_conversations": 0
            }

    def add_to_conversation(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        """Add new messages to a conversation with proper metadata."""
        timestamp = datetime.now()
        sequence_base = len(self.get_conversation_history(conversation_id)) * 1000
        
        for i, message in enumerate(messages):
            # Create embedding for the message
            text = message['text']
            vector = np.array([0.0] * 1536)  # Placeholder until we get real embeddings
            
            # Store the message
            self.add_vector(
                key=f"vector:chat:{conversation_id}_{sequence_base + i}",
                vector=vector.tolist(),
                metadata={
                    "conversation_id": conversation_id,
                    "timestamp": timestamp.isoformat(),
                    "sequence": sequence_base + i,
                    "text": text,
                    "role": message['role'],
                    "chunk_index": 0,
                    "message_index": sequence_base + i,
                    "used_in_context": False
                }
            )

    def add_vector_to_pipeline(self, pipeline, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        """Add vector and metadata to Redis pipeline for batch processing."""
        # Convert vector to bytes for storage
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        # Add to pipeline
        pipeline.hset(f"vector:{key}", "embedding", vector_bytes)
        pipeline.hset(f"vector:{key}", "metadata", json.dumps(metadata))
        
    def add_vectors_batch(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Add multiple vectors in a single batch operation."""
        pipeline = self.redis_client.pipeline()
        
        for key, vector, metadata in vectors:
            self.add_vector_to_pipeline(pipeline, key, vector, metadata)
            
        pipeline.execute()

    def get_memory(self, conversation_id: str, sequence: int) -> Optional[Dict[str, Any]]:
        """Get a specific memory by conversation ID and sequence number."""
        try:
            key = f"vector:chat:{conversation_id}_{sequence}"
            result = self.redis_client.hget(key, "metadata")
            if result:
                return json.loads(result.decode())
            return None
        except Exception as e:
            print(f"Error retrieving memory {conversation_id}:{sequence}: {str(e)}")
            return None