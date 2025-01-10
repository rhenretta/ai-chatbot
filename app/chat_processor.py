import openai
from typing import List, Dict, Any
from .database import VectorStore
from .embeddings import EmbeddingsManager

class ChatProcessor:
    def __init__(self, api_key: str, vector_store: VectorStore, embeddings_manager: EmbeddingsManager):
        openai.api_key = api_key
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.model = "gpt-3.5-turbo"
    
    def _format_context(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        
        context = "Here are some relevant memories from our past conversations:\n\n"
        for memory in memories:
            context += f"Memory from {memory.get('timestamp', 'unknown time')}:\n{memory.get('text', '')}\n\n"
        return context
    
    def process_message(self, message: str, max_memories: int = 3) -> str:
        # Create embedding for the query
        query_embedding = self.embeddings_manager.create_embedding(message)
        
        # Search for similar memories
        similar_memories = self.vector_store.search_similar(query_embedding, top_k=max_memories)
        
        # Extract metadata from results
        memories = [memory[2] for memory in similar_memories if memory[1] > 0.7]  # Only use memories with high similarity
        
        # Format context with memories
        context = self._format_context(memories)
        
        # Create the chat completion
        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"You are a helpful AI assistant with access to memories from past conversations. {context}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant."
            })
        
        messages.append({
            "role": "user",
            "content": message
        })
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
