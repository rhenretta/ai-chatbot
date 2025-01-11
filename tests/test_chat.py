import pytest
import json
import asyncio
from datetime import datetime, timedelta
from app.chat_processor import ChatProcessor, ConversationMemoryAgent
from app.database import VectorStore
from app.embeddings import EmbeddingsManager
from app.models import ProcessingStats

# Test data - A conversation about naming discussions
SAMPLE_CONVERSATION = {
    "id": "test_conversation_1",
    "mapping": {
        "msg1": {
            "message": {
                "content": {
                    "parts": ["What should we name our AI assistant?"]
                },
                "author": {"role": "user"},
                "create_time": (datetime.now() - timedelta(days=1)).timestamp()
            }
        },
        "msg2": {
            "message": {
                "content": {
                    "parts": ["I've been thinking about some options: 'Mnemosyne' (Greek goddess of memory), 'Echo' (for conversation recall), or 'Sage' (for wisdom). What do you think?"]
                },
                "author": {"role": "assistant"},
                "create_time": (datetime.now() - timedelta(days=1)).timestamp()
            }
        },
        "msg3": {
            "message": {
                "content": {
                    "parts": ["Those are interesting. What about something more modern like 'Cortex' or 'Neural'?"]
                },
                "author": {"role": "user"},
                "create_time": (datetime.now() - timedelta(days=1)).timestamp()
            }
        },
        "msg4": {
            "message": {
                "content": {
                    "parts": ["'Whatsername' could be fun too - it's ironic since the AI remembers everything!"]
                },
                "author": {"role": "assistant"},
                "create_time": (datetime.now() - timedelta(days=1)).timestamp()
            }
        },
        "msg5": {
            "message": {
                "content": {
                    "parts": ["Haha, I like that! Let's go with Whatsername. It's perfect since she'll remember everything while having a name that suggests forgetfulness."]
                },
                "author": {"role": "user"},
                "create_time": (datetime.now() - timedelta(days=1)).timestamp()
            }
        }
    }
}

@pytest.fixture
async def chat_setup():
    """Setup test environment with vector store and chat processor."""
    # Initialize components with test configurations
    vector_store = VectorStore(
        host="localhost",
        port=6379,
        index_name="test_memory_index"
    )
    
    embeddings_manager = EmbeddingsManager()
    
    chat_processor = ChatProcessor(
        api_key="test_key",
        vector_store=vector_store,
        embeddings_manager=embeddings_manager
    )
    
    # Clear any existing test data
    await vector_store.clear_all()
    
    # Process the sample conversation
    stats = ProcessingStats()
    await chat_processor.process_conversation_upload(SAMPLE_CONVERSATION, stats)
    
    yield chat_processor
    
    # Cleanup
    await vector_store.clear_all()

@pytest.mark.asyncio
async def test_memory_retrieval(chat_setup):
    """Test the chat memory retrieval functionality."""
    chat_processor = chat_setup
    
    # Test query about name discussion
    response1 = await chat_processor.process_message(
        "what are some alternate names we discussed when coming up with the name for whatsername?"
    )
    
    # Verify response contains the alternative names
    assert any(name in response1["response"].lower() for name in ["mnemosyne", "echo", "sage", "cortex", "neural"])
    assert response1["context_used"] == True
    assert "search_conversations" in response1["tools_used"]
    
    # Test follow-up request for conversation
    response2 = await chat_processor.process_message(
        "I don't remember that, show me the conversation"
    )
    
    # Verify response contains conversation snippets
    assert "what should we name our ai assistant?" in response2["response"].lower()
    assert response2["context_used"] == True
    assert "search_conversations" in response2["tools_used"]

@pytest.mark.asyncio
async def test_conversation_processing(chat_setup):
    """Test the conversation processing functionality."""
    chat_processor = chat_setup
    
    # Verify stats after processing
    stats = await chat_processor.vector_store.get_memory_stats()
    assert stats["total_memories"] > 0
    
    # Test memory persistence
    response = await chat_processor.process_message(
        "When did we decide on the name Whatsername?"
    )
    assert "ironic" in response["response"].lower()
    assert response["context_used"] == True

@pytest.mark.asyncio
async def test_error_handling(chat_setup):
    """Test error handling in chat processing."""
    chat_processor = chat_setup
    
    # Test with empty message
    response = await chat_processor.process_message("")
    assert "error" in response
    
    # Test with very long message
    long_message = "test " * 1000
    response = await chat_processor.process_message(long_message)
    assert response["response"] != ""
    
if __name__ == "__main__":
    pytest.main(["-v", "test_chat.py"]) 