import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import VectorStore
from app.embeddings import EmbeddingsManager
from app.chat_processor import ChatProcessor
from dotenv import load_dotenv
import json
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_pipeline():
    load_dotenv()
    
    # Initialize components
    vector_store = VectorStore()
    embeddings_manager = EmbeddingsManager(os.getenv("OPENAI_API_KEY"))
    chat_processor = ChatProcessor(
        os.getenv("OPENAI_API_KEY"),
        vector_store,
        embeddings_manager
    )
    
    # Test data
    test_message = "What is machine learning?"
    
    try:
        # Test embedding creation
        logger.debug("Testing embedding creation...")
        embedding = embeddings_manager.create_embedding(test_message)
        logger.debug(f"Embedding dimension: {len(embedding)}")
        
        # Test vector store
        logger.debug("Testing vector store...")
        vector_store.add_vector(
            "test_key",
            embedding,
            {"text": test_message, "timestamp": "2024-03-20T10:00:00Z"}
        )
        
        # Test similarity search
        logger.debug("Testing similarity search...")
        similar = vector_store.search_similar(embedding, top_k=1)
        logger.debug(f"Similar results: {similar}")
        
        # Test chat processing
        logger.debug("Testing chat processing...")
        response = chat_processor.process_message(test_message)
        logger.debug(f"Chat response: {response}")
        
    except Exception as e:
        logger.error(f"Error during debugging: {str(e)}", exc_info=True)

if __name__ == "__main__":
    debug_pipeline() 