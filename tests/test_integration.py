import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from app.main import app
from tests.config import TEST_CONFIG
import json

client = TestClient(app)

@pytest.fixture
def mock_embedding_response():
    return Mock(data=[Mock(embedding=[0.1, 0.2, 0.3, 0.4])])

@pytest.fixture
def mock_chat_response():
    return Mock(choices=[Mock(message=Mock(content="This is a mock response"))])

@pytest.fixture(autouse=True)
def mock_redis():
    with patch('app.database.redis.Redis') as mock:
        # Mock the Redis client methods
        mock_client = Mock()
        mock_client.pipeline.return_value = Mock()
        mock_client.pipeline.return_value.execute.return_value = None
        mock_client.hgetall.return_value = {
            b"embedding": bytes([0] * 16),  # Mock embedding bytes
            b"metadata": json.dumps({"text": "mock text"}).encode()
        }
        mock_client.keys.return_value = [b"vector:test_key"]
        mock.return_value = mock_client
        yield mock

def test_chat_endpoint(mock_embedding_response, mock_chat_response):
    with patch('openai.resources.embeddings.Embeddings.create', return_value=mock_embedding_response), \
         patch('openai.resources.chat.completions.Completions.create', return_value=mock_chat_response):
        
        test_message = "What is machine learning?"
        response = client.post(
            "/chat",
            json={"message": test_message}
        )
        
        assert response.status_code == 200
        assert "response" in response.json()
        assert isinstance(response.json()["response"], str)
        assert response.json()["response"] == "This is a mock response"

def test_upload_and_query(mock_embedding_response, mock_chat_response):
    with patch('openai.resources.embeddings.Embeddings.create', return_value=mock_embedding_response), \
         patch('openai.resources.chat.completions.Completions.create', return_value=mock_chat_response):
        
        # Test file upload
        test_data = json.dumps(TEST_CONFIG["TEST_CONVERSATION_DATA"])
        response = client.post(
            "/upload",
            files={"file": ("test.json", test_data)}
        )
        
        assert response.status_code == 200
        
        # Test querying the uploaded data
        query_response = client.post(
            "/chat",
            json={"message": "What did we discuss about machine learning?"}
        )
        
        assert query_response.status_code == 200
        assert "response" in query_response.json()
        assert query_response.json()["response"] == "This is a mock response" 