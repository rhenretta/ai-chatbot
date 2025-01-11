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

# Enhanced Agent Flow Documentation

## Overview
The ConversationMemoryAgent uses a multi-stage reasoning and memory retrieval system to process complex queries, understand context, and generate insightful responses that draw from past conversations intelligently.

## Core Components
1. Vector Store: Stores embeddings of past conversations with metadata
2. Embeddings Manager: Handles creation and management of embeddings
3. LLM: GPT-4 for advanced reasoning and response generation
4. Memory Manager: Handles intelligent memory retrieval and synthesis
5. State Graph: Orchestrates the multi-stage workflow

## Enhanced Message Processing Flow

1. Query Analysis Stage
   - Analyzes input message for:
     - Type of query (reflection, factual, opinion, etc.)
     - Required memory types (emotional, technical, behavioral)
     - Temporal aspects (recent vs historical)
     - Need for synthesis vs specific details
   - Outputs a reasoning plan with:
     - Memory retrieval strategy
     - Required search perspectives
     - Response synthesis approach

2. Intelligent Memory Retrieval
   - Multi-perspective Search:
     - Direct semantic search based on query
     - Inferred semantic searches based on query analysis
     - Temporal pattern matching
     - Emotional/behavioral pattern matching
   - Memory Synthesis:
     - Clusters similar memories
     - Identifies patterns and trends
     - Extracts key insights
     - Maintains source traceability

3. Context Building
   - Combines multiple types of context:
     - Direct conversation snippets
     - Synthesized insights
     - Behavioral patterns
     - Emotional patterns
     - User preferences and tendencies
   - Builds a comprehensive user model:
     - Interaction patterns
     - Consistent traits
     - Evolution over time
     - Areas of interest/expertise

4. Response Generation
   - Multi-stage response building:
     - High-level insights synthesis
     - Supporting evidence selection
     - Detail level adjustment
     - Source reference management
   - Response characteristics:
     - Includes synthesized insights
     - References specific conversations when relevant
     - Maintains ability to drill down into sources
     - Adapts detail level to query type

5. Memory Management
   - Intelligent Storage:
     - Stores both raw conversations and synthesized insights
     - Maintains relationship graphs between memories
     - Tracks memory usage patterns
     - Updates user model continuously
   - Context Persistence:
     - Maintains conversation thread understanding
     - Tracks insight evolution
     - Manages memory relevance scoring
     - Updates memory importance weights

## Example Query Processing
For a query like "based on our deepest, most meaningful chats, how would you describe me":

1. Query Analysis:
   - Type: Reflective, personality analysis
   - Required memories: Emotional, behavioral, philosophical discussions
   - Need for synthesis: High
   - Temporal scope: All-time

2. Memory Retrieval:
   - Search perspectives:
     - Emotional depth indicators
     - Personal revelations
     - Philosophical discussions
     - Behavioral patterns
     - User's self-reflection moments

3. Synthesis:
   - Pattern recognition across conversations
   - Personality trait extraction
   - Growth/change tracking
   - Consistent characteristics identification

4. Response Formation:
   - High-level personality insights
   - Supporting conversation references
   - Notable growth patterns
   - Specific memorable interactions

## Implementation Notes
- Uses GPT-4 for advanced reasoning
- Implements memory relationship graphs
- Maintains user model updates
- Tracks conversation importance metrics
- Enables drill-down capability for details
- Supports dynamic context adjustment
