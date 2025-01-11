import os
from openai import OpenAI
from typing import List, Dict, Any
import numpy as np

class EmbeddingsManager:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-ada-002"
        self.max_tokens = 8191  # OpenAI's limit
        self.batch_size = 20  # Process 20 texts at a time
        
    def create_chunks(self, text: str) -> List[str]:
        """Create chunks from text that fit within token limits."""
        # Rough token estimation (words * 1.3)
        words = text.split()
        estimated_tokens = len(words) * 1.3
        
        if estimated_tokens <= self.max_tokens:
            return [text]
            
        # Split into chunks based on estimated token count
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = len(word) * 1.3
            if current_tokens + word_tokens > self.max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
        
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in one API call."""
        if not texts:
            return []
            
        # Process in batches to avoid API limits
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)
            
        return all_embeddings
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embeddings = self.get_embeddings_batch([text])
        return embeddings[0] if embeddings else []
