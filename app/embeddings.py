import os
from openai import OpenAI
from typing import List, Dict, Any

class EmbeddingsManager:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def create_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def process_chatgpt_export(self, conversation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        messages = conversation_data.get("messages", [])
        
        for message in messages:
            chunk = {
                "text": message["text"],
                "conversation_id": conversation_data["conversation_id"],
                "role": message["role"],
                "timestamp": message["timestamp"],
                "embedding": self.create_embedding(message["text"])
            }
            chunks.append(chunk)
            
        return chunks
