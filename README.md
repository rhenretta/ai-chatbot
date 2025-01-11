# AI Memory Chatbot

A FastAPI-based chatbot application that processes and manages conversations with vector-based memory storage and real-time progress tracking.

## Features

- **File Processing**
  - Support for ZIP and JSON file uploads
  - Real-time upload progress tracking
  - Batch conversation processing
  - Processing statistics (success, skip, error counts)

- **Memory Management**
  - Vector-based conversation storage using Redis
  - Efficient conversation retrieval
  - Memory usage statistics
  - Conversation history tracking

- **User Interface**
  - Real-time progress updates during uploads
  - Memory statistics display
  - Conversation history viewer
  - Tool usage transparency

## Requirements

- Python 3.9+
- Redis server
- OpenAI API key
- AWS Region: us-east-2

## Setup

1. Clone the repository
2. Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Redis server
5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

- `GET /`: Main chat interface
- `POST /chat`: Process chat messages
- `POST /upload`: Upload and process conversation files
- `GET /conversation/{conversation_id}`: Retrieve conversation history
- `GET /stats`: Get memory statistics

## File Upload Format

Supports two formats:
1. Single JSON file containing conversation data
2. ZIP file containing `conversations.json`

## Development

- Uses FastAPI for the web framework
- SSE (Server-Sent Events) for real-time progress updates
- Redis for vector storage
- OpenAI for embeddings and chat processing
- Type hints and documentation throughout the codebase

## Testing

Run tests with:
```bash
pytest tests/
```

## Security

- API keys stored in environment variables
- Input sanitization
- File upload validation
- Rate limiting on API requests
