import pytest
from unittest.mock import Mock, patch
from app.embeddings import EmbeddingsManager
from tests.config import TEST_CONFIG

@pytest.fixture
def mock_embedding_response():
    return Mock(data=[Mock(embedding=[0.1, 0.2, 0.3, 0.4])])

@pytest.fixture
def embeddings_manager():
    return EmbeddingsManager("fake_api_key")

def test_create_embedding(embeddings_manager, mock_embedding_response):
    with patch.object(embeddings_manager.client.embeddings, 'create', return_value=mock_embedding_response):
        test_text = "This is a test message"
        embedding = embeddings_manager.create_embedding(test_text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        
        # Verify the API was called with correct parameters
        embeddings_manager.client.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=test_text
        )

def test_process_chatgpt_export(embeddings_manager, mock_embedding_response):
    with patch.object(embeddings_manager.client.embeddings, 'create', return_value=mock_embedding_response):
        test_data = TEST_CONFIG["TEST_CONVERSATION_DATA"]
        chunks = embeddings_manager.process_chatgpt_export(test_data)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("text" in chunk for chunk in chunks)
        assert all("conversation_id" in chunk for chunk in chunks)
        assert all("embedding" in chunk for chunk in chunks)
        
        # Verify the API was called for each message
        assert embeddings_manager.client.embeddings.create.call_count == len(test_data["messages"]) 