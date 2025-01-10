import redis
import numpy as np
import json
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.prefix = "memory:"
        self.metadata_prefix = "metadata:"
    
    def add_vector(self, key: str, vector: np.ndarray, metadata: Dict[str, Any]):
        vector_key = f"{self.prefix}{key}"
        metadata_key = f"{self.metadata_prefix}{key}"
        
        # Store the vector
        vector_bytes = vector.tobytes()
        self.redis_client.set(vector_key, vector_bytes)
        
        # Store the metadata
        self.redis_client.set(metadata_key, json.dumps(metadata))
    
    def get_vector(self, key: str) -> tuple[np.ndarray, Dict[str, Any]]:
        vector_key = f"{self.prefix}{key}"
        metadata_key = f"{self.metadata_prefix}{key}"
        
        # Get vector
        vector_bytes = self.redis_client.get(vector_key)
        if vector_bytes is None:
            raise KeyError(f"No vector found for key: {key}")
        
        vector = np.frombuffer(vector_bytes, dtype=np.float32)
        
        # Get metadata
        metadata_str = self.redis_client.get(metadata_key)
        metadata = json.loads(metadata_str) if metadata_str else {}
        
        return vector, metadata
    
    def search_similar(self, query_vector: np.ndarray, top_k: int = 5) -> List[tuple[str, float, Dict[str, Any]]]:
        results = []
        
        # Get all keys
        vector_keys = self.redis_client.keys(f"{self.prefix}*")
        
        for key in vector_keys:
            key_str = key.decode('utf-8').replace(self.prefix, "")
            vector, metadata = self.get_vector(key_str)
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((key_str, float(similarity), metadata))
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def clear_all(self):
        vector_keys = self.redis_client.keys(f"{self.prefix}*")
        metadata_keys = self.redis_client.keys(f"{self.metadata_prefix}*")
        
        for key in vector_keys + metadata_keys:
            self.redis_client.delete(key)
