from typing import List, Dict, Any
from openai import OpenAI
from .database import VectorStore
from .embeddings import EmbeddingsManager

class ChatProcessor:
    def __init__(self, api_key: str, vector_store: VectorStore, embeddings_manager: EmbeddingsManager):
        self.client = OpenAI(api_key=api_key)
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        
    def process_message(self, message: str, top_k: int = 3) -> str:
        # Create embedding for the message
        query_embedding = self.embeddings_manager.create_embedding(message)
        
        # Retrieve similar messages from vector store
        similar_messages = self.vector_store.search_similar(query_embedding, top_k=top_k)
        
        # Construct the prompt with context
        context = "\n".join([f"Previous conversation: {msg['text']}" for msg in similar_messages])
        prompt = f"Context from previous conversations:\n{context}\n\nCurrent message: {message}\n\nResponse:"
        
        # Generate response using ChatGPT
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with memory of previous conversations."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
