# AI Memory RAG System

A proof of concept system that allows users to upload their ChatGPT conversation history and interact with an AI interface enhanced with retrieval-augmented generation (RAG) to simulate memory and lifelike responses.

## Features

- Upload and process ChatGPT conversation exports
- Vector-based conversation memory storage using Redis
- Retrieval-augmented generation for contextual responses
- Simple web interface for chat interactions

## Prerequisites

- Python 3.8+
- Redis server
- OpenAI API key

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env`

4. Start Redis server:
   ```bash
   redis-server
   ```

5. Run the application:
   ```bash
   python -m uvicorn app.main:app --reload
   ```

## Usage

1. Access the web interface at `http://localhost:8000`
2. Upload your ChatGPT conversation export file (JSON format)
3. Start chatting with the AI, which will now have context from your past conversations

## Architecture

- FastAPI backend with Redis vector store
- OpenAI's text-embedding-ada-002 for embeddings
- ChatGPT API for response generation
- Simple HTML/JS frontend

## Future Enhancements

- Improved memory retrieval algorithms
- Conversation persistence
- User authentication
- Advanced conversation chunking strategies
- Support for multiple conversation history formats

## License

MIT License
