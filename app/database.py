import redis
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        
    def add_vector(self, key: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        # Convert vector to bytes for storage
        vector_bytes = np.array(vector, dtype=np.float32).tobytes()
        
        # Store the vector and metadata
        pipeline = self.redis_client.pipeline()
        pipeline.hset(f"vector:{key}", "embedding", vector_bytes)
        pipeline.hset(f"vector:{key}", "metadata", json.dumps(metadata))
        pipeline.execute()
        
    def update_memory_metadata(self, conversation_id: str, sequence: int, metadata: Dict[str, Any]) -> None:
        key = f"conv:{conversation_id}:{sequence}"
        result = self.redis_client.hget(f"vector:{key}", "metadata")
        if result:
            current_metadata = json.loads(result)
            current_metadata.update(metadata)
            self.redis_client.hset(f"vector:{key}", "metadata", json.dumps(current_metadata))

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get the full history of a conversation."""
        if not conversation_id:
            return []
            
        try:
            # Get all keys for this conversation
            pattern = f"conv:{conversation_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return []
                
            # Get all messages with their metadata
            pipeline = self.redis_client.pipeline()
            for key in keys:
                pipeline.hgetall(key)
            results = pipeline.execute()
            
            # Filter and sort messages
            messages = []
            for result in results:
                if result and 'metadata' in result:
                    try:
                        metadata = json.loads(result['metadata'])
                        if metadata.get('text'):  # Only include messages with text
                            messages.append({
                                'conversation_id': metadata.get('conversation_id'),
                                'timestamp': metadata.get('timestamp'),
                                'sequence': metadata.get('sequence', 0),
                                'text': metadata.get('text'),
                                'role': metadata.get('role', 'unknown')
                            })
                    except json.JSONDecodeError:
                        continue
            
            # Sort by sequence number to maintain order
            return sorted(messages, key=lambda x: x['sequence'])
            
        except Exception as e:
            print(f"Error getting conversation history: {str(e)}")
            return []

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all stored memories/chunks across all conversations."""
        keys = self.redis_client.keys("vector:conv:*")
        memories = []
        for key in keys:
            result = self.redis_client.hget(key, "metadata")
            if result:
                memories.append(json.loads(result))
        return memories

    def search_similar(self, query_vector: List[float], top_k: int = 5, conversation_id: Optional[str] = None, min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar vectors with improved error handling and accuracy."""
        try:
            # Convert query vector to numpy for calculations
            query_np = np.array(query_vector, dtype=np.float32)
            
            # Get all vectors
            pattern = f"vector:{'conv:' + conversation_id + ':*' if conversation_id else 'conv:*'}"
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
                    metadata = json.loads(result[b"metadata"])
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_np, vector) / (np.linalg.norm(query_np) * np.linalg.norm(vector))
                    
                    # Only include results above similarity threshold
                    if similarity >= min_similarity:
                        results.append((similarity, metadata))
                        
                except Exception as e:
                    print(f"Error processing vector {key}: {str(e)}")
                    continue
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[0], reverse=True)
            return [r[1] for r in results[:top_k]]
            
        except Exception as e:
            print(f"Error in search_similar: {str(e)}")
            return []

    def get_vector(self, key: str) -> tuple[List[float], Dict[str, Any]]:
        result = self.redis_client.hgetall(f"vector:{key}")
        
        if not result:
            raise KeyError(f"No vector found for key: {key}")
            
        vector = np.frombuffer(result[b"embedding"], dtype=np.float32).tolist()
        metadata = json.loads(result[b"metadata"])
        return vector, metadata
        
    def vector_exists(self, key: str) -> bool:
        """Check if a vector already exists in Redis."""
        return self.redis_client.exists(f"vector:{key}")
    
    def get_existing_conversation_ids(self) -> set:
        """Get all unique conversation IDs from Redis."""
        keys = self.redis_client.keys("vector:conv:*")
        conversation_ids = set()
        for key in keys:
            try:
                metadata = json.loads(self.redis_client.hget(key, "metadata"))
                conversation_ids.add(metadata.get("conversation_id"))
            except:
                continue
        return conversation_ids
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about stored memories."""
        memories = self.get_all_memories()
        return {
            "total_memories": len(memories),
            "used_in_context": len([m for m in memories if m.get("used_in_context", False)]),
            "conversation_count": len(self.get_existing_conversation_ids())
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
                key=f"conv:{conversation_id}:{sequence_base + i}",
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
            key = f"vector:conv:{conversation_id}:{sequence}"
            result = self.redis_client.hget(key, "metadata")
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            print(f"Error retrieving memory {conversation_id}:{sequence}: {str(e)}")
            return None