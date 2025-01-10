import openai
import numpy as np
from typing import List, Dict, Any
import tiktoken
import json
from datetime import datetime

class EmbeddingsManager:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.model = "text-embedding-ada-002"
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 8191  # OpenAI's limit for text-embedding-ada-002
    
    def create_embedding(self, text: str) -> np.ndarray:
        response = openai.Embedding.create(
            input=text,
            model=self.model
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    
    def chunk_conversation(self, conversation: Dict[str, Any], chunk_size: int = 500) -> List[Dict[str, Any]]:
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        messages = conversation.get('messages', [])
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if not content:
                continue
            
            tokens = self.encoding.encode(content)
            token_count = len(tokens)
            
            if current_tokens + token_count > chunk_size:
                # Create chunk
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'conversation_id': conversation.get('id'),
                    'timestamp': conversation.get('create_time'),
                    'chunk_index': len(chunks)
                })
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(f"{role}: {content}")
            current_tokens += token_count
        
        # Add remaining chunk if any
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'conversation_id': conversation.get('id'),
                'timestamp': conversation.get('create_time'),
                'chunk_index': len(chunks)
            })
        
        return chunks
    
    def process_chatgpt_export(self, file_content: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(file_content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in the export file")
        
        all_chunks = []
        for conversation in data:
            chunks = self.chunk_conversation(conversation)
            all_chunks.extend(chunks)
        
        return all_chunks
