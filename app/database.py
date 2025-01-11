import redis
import json
from typing import List, Dict, Any
import numpy as np

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
        
    def get_vector(self, key: str) -> tuple[List[float], Dict[str, Any]]:
        # Get vector and metadata from Redis
        result = self.redis_client.hgetall(f"vector:{key}")
        
        if not result:
            raise KeyError(f"No vector found for key: {key}")
            
        # Convert bytes back to vector
        vector = np.frombuffer(result[b"embedding"], dtype=np.float32).tolist()
        metadata = json.loads(result[b"metadata"])
        
        return vector, metadata
        
    def search_similar(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        # Convert query vector to numpy array
        query_array = np.array(query_vector, dtype=np.float32)
        
        # Get all keys
        keys = [key.decode('utf-8').split(':')[1] for key in self.redis_client.keys("vector:*")]
        
        results = []
        for key in keys:
            vector, metadata = self.get_vector(key)
            similarity = np.dot(query_array, vector) / (np.linalg.norm(query_array) * np.linalg.norm(vector))
            results.append((key, similarity, metadata))
        
        # Sort by similarity and return top_k results
        results.sort(key=lambda x: x[1], reverse=True)
        return [result[2] for result in results[:top_k]]
