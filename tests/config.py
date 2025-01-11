import os
from dotenv import load_dotenv

load_dotenv()

TEST_CONFIG = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
    "REDIS_PORT": int(os.getenv("REDIS_PORT", 6379)),
    "TEST_CONVERSATION_DATA": {
        "conversation_id": "test_convo_1",
        "messages": [
            {
                "text": "What is machine learning?",
                "role": "user",
                "timestamp": "2024-03-20T10:00:00Z"
            },
            {
                "text": "Machine learning is a subset of artificial intelligence...",
                "role": "assistant",
                "timestamp": "2024-03-20T10:00:05Z"
            }
        ]
    }
} 