# AI Memory Chatbot with RAG

A powerful chatbot that maintains context and memory across conversations using Retrieval-Augmented Generation (RAG).

## Features

- 📝 Conversation Memory
  - Maintains context across multiple conversations
  - Natural time-based references
  - Semantic search for relevant memories
  - Batch processing for efficient storage

- 🔄 Import/Export
  - ChatGPT conversation export support
  - ZIP file handling
  - Progress tracking for uploads
  - Duplicate prevention

- 🧠 Smart Context
  - Real-time memory retrieval
  - Relevance-based context building
  - Natural conversation references
  - Full conversation history access

- ⚡ Performance
  - Redis vector store
  - Batch embeddings processing
  - Optimized similarity search
  - Efficient chunking

## Getting Started

### Prerequisites

- Python 3.9+
- Redis server
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-chatbot.git
cd ai-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

4. Start Redis server:
```bash
redis-server
```

5. Run the application:
```bash
uvicorn app.main:app --reload
```

## Usage

### Chat Interface

Visit `http://localhost:8000` to access the chat interface. Features include:
- Real-time conversation
- Memory statistics
- Context display
- File upload

### API Endpoints

- `POST /chat`: Send messages and get responses
- `POST /upload`: Upload conversation history
- `GET /conversation/{id}`: Get conversation details
- `GET /stats`: Get memory statistics

## Architecture

### Components

1. ChatProcessor
   - Handles message processing
   - Manages RAG integration
   - Tracks conversation context

2. VectorStore
   - Redis-based storage
   - Vector similarity search
   - Metadata management

3. EmbeddingsManager
   - OpenAI embeddings generation
   - Token management
   - Batch processing

### Technologies

- Backend: FastAPI
- Vector Store: Redis
- Embeddings: OpenAI (text-embedding-ada-002)
- AI: GPT-3.5-turbo
- Frontend: HTML/JS with Tailwind CSS

## Development

### Guidelines

1. Code Style
   - Use type hints
   - Document functions
   - Follow PEP 8

2. Testing
   - Test various file formats
   - Verify memory retrieval
   - Check context accuracy

3. Security
   - Use environment variables
   - Implement rate limiting
   - Sanitize inputs

### Current Status

✓ Core Features
- Chat interface
- Memory management
- Context retrieval
- File processing

🔄 In Progress
- Performance optimization
- Enhanced error handling
- Extended testing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT and embeddings APIs
- FastAPI for the web framework
- Redis for vector storage
