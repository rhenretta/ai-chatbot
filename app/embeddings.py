import os
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Any, Optional
import numpy as np
import asyncio

class EmbeddingsManager:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
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
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text string."""
        try:
            if not isinstance(text, str):
                print(f"Invalid text type: {type(text)}")
                return None
                
            if not text.strip():
                print("Empty text provided")
                return None
                
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return None

    async def get_embedding_async(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text string asynchronously."""
        try:
            if not isinstance(text, str):
                print(f"Invalid text type: {type(text)}")
                return None
                
            if not text.strip():
                print("Empty text provided")
                return None
                
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding async: {str(e)}")
            return None

    async def get_embeddings_batch_async(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts in parallel."""
        try:
            if not texts:
                print("Empty texts list provided")
                return []
                
            tasks = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tasks.append(self.async_client.embeddings.create(
                    model=self.model,
                    input=batch
                ))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            embeddings = []
            
            for response in responses:
                if isinstance(response, Exception):
                    print(f"Error in batch processing: {str(response)}")
                    embeddings.extend([None] * self.batch_size)
                else:
                    embeddings.extend([data.embedding for data in response.data])
            
            return embeddings
        except Exception as e:
            print(f"Error in batch embeddings: {str(e)}")
            return [None] * len(texts)
