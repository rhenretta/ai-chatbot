from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict, Annotated
from datetime import datetime
import json
import uuid
import re
import asyncio
from operator import itemgetter
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Redis
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from .database import VectorStore
from .embeddings import EmbeddingsManager
from .models import ProcessingStats

MAX_TOKENS = 6000  # Increased from 4000, still safe for text-embedding-ada-002 (max 8191)
CHUNK_OVERLAP = 200  # Number of tokens to overlap between chunks
MESSAGE_CHUNK_SIZE = 10  # Increased from 5 to reduce vector store fragmentation

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks that don't exceed max_tokens."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        # Find end of current chunk
        end = start + max_tokens
        if end >= len(tokens):
            chunks.append(encoding.decode(tokens[start:]))
            break
            
        # Find a good breaking point
        # Look for newline or period in overlap region
        overlap_start = max(start + max_tokens - overlap, 0)
        overlap_tokens = tokens[overlap_start:end]
        overlap_text = encoding.decode(overlap_tokens)
        
        # Try to break at newline first, then period
        break_at = overlap_text.rfind('\n')
        if break_at == -1:
            break_at = overlap_text.rfind('.')
            if break_at == -1:
                break_at = len(overlap_text) - 1
                
        chunk_end = overlap_start + break_at + 1
        chunks.append(encoding.decode(tokens[start:chunk_end]))
        start = chunk_end
        
    return chunks

class AgentState(TypedDict):
    """State of the agent during execution."""
    messages: List[BaseMessage]
    next_step: str
    current_input: str
    context: List[Dict[str, Any]]
    tools_used: List[str]
    final_answer: Optional[str]

class ConversationMemoryAgent:
    def __init__(self, vector_store: VectorStore, embeddings_manager: EmbeddingsManager, api_key: str):
        """Initialize the conversation memory agent."""
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.api_key = api_key
        self.chat_history = ChatMessageHistory()
        self.active_memories = {}  # Store recently used memories with timestamps
        self.last_mentioned_conversation = None  # Track the last mentioned conversation
        
        # Create the LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo-16k",
            openai_api_key=self.api_key,
            max_tokens=1000  # Limit response length
        )
        
        # Define tools
        self.tools = [
            Tool(
                name="search_conversations",
                description="Search for relevant past conversations",
                func=self._search_conversations
            )
        ]
        
        self.tool_executor = ToolExecutor(self.tools)
        
        # Create the prompts
        self.decide_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with memory access. Based on the conversation history and current input, decide what to do next:

1. If the user explicitly asks about past conversations or memories, use 'search' to look for relevant context
2. If you have enough context to respond, use 'respond'

Output your decision as a simple string: either 'search' or 'respond'."""),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ])
        
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", "Based on the current conversation, formulate a search query to find relevant past conversations."),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "Current input: {input}\nCurrent context: {context}")
        ])
        
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with access to conversation history. Use the provided context to give informed responses.
If the context contains relevant information from past conversations, incorporate it naturally into your response.

Important guidelines:
1. NEVER say "as an AI" or mention limitations about emotions/preferences
2. When asked about favorite conversations or memories:
   - Search the context for interesting or unique exchanges
   - Share specific examples with proper formatting
   - Focus on the content and what made it interesting
3. When showing conversation snippets:
   - Only show the exact conversation being referenced
   - Format it in a clear, readable way with proper spacing
   - Use "Me:" for user messages and "You:" for assistant messages
4. Never use phrases like "give me a moment" or similar that imply another response is coming
5. If you can't find relevant information in the context, focus on the current conversation instead of disclaimers

When displaying conversation snippets, format them like this:

Me: [user's message]

You: [assistant's message]"""),
            ("system", "Context from memory: {context}"),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{input}")
        ])
        
        # Create the graph
        print("Initializing workflow graph...")
        self.workflow = self._create_graph()
        print("Workflow graph initialized successfully")
        
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Define the nodes
        
        # Decision node
        def decide(state: AgentState) -> Dict[str, Any]:
            messages = state["messages"]
            current_input = state["current_input"]
            context = state["context"]
            tools_used = state["tools_used"]
            
            try:
                # First, automatically check for relevant memories
                results = self._search_conversations(current_input)
                
                if results:
                    print(f"Found {len(results)} initial relevant memories")
                    context = context + results
                    tools_used = tools_used + ["search_conversations"]
                
                # Now decide if we need to explicitly search or can respond
                response = self.decide_prompt | self.llm
                result = response.invoke({
                    "messages": messages,
                    "input": current_input
                })
                
                # Extract decision from response
                response_text = result.content if hasattr(result, 'content') else str(result)
                print(f"Decision response: {response_text}")
                
                # Clean and normalize the response
                decision = response_text.strip().lower()
                if "search" in decision and not results:  # Only search if we haven't found results yet
                    print("Deciding to perform explicit search")
                    return {
                        "messages": messages,
                        "current_input": current_input,
                        "context": context,
                        "tools_used": tools_used,
                        "next_step": "search"
                    }
                
                print("Deciding to respond with current context")
                return {
                    "messages": messages,
                    "current_input": current_input,
                    "context": context,
                    "tools_used": tools_used,
                    "next_step": "respond"
                }
                
            except Exception as e:
                print(f"Error in decide node: {str(e)}, defaulting to respond")
                return {
                    "messages": messages,
                    "current_input": current_input,
                    "context": context,
                    "tools_used": tools_used,
                    "next_step": "respond"
                }
        
        # Search node
        def search(state: AgentState) -> Dict[str, Any]:
            messages = state["messages"]
            current_input = state["current_input"]
            context = state["context"]
            tools_used = state["tools_used"]
            
            try:
                # Generate search query
                response = (self.search_prompt | self.llm).invoke({
                    "messages": messages,
                    "input": current_input,
                    "context": context
                })
                
                # Extract query text from response
                query = response.content if hasattr(response, 'content') else str(response)
                print(f"Generated search query: {query}")
                
                # Execute search with the string query
                results = self._search_conversations(query)
                
                # Update state and go directly to respond to prevent recursion
                return {
                    "messages": messages,
                    "current_input": current_input,
                    "context": context + results,
                    "tools_used": tools_used + ["search_conversations"],
                    "next_step": "respond"  # Go directly to respond instead of decide
                }
            except Exception as e:
                print(f"Error in search node: {str(e)}")
                # On error, go to respond with current context
                return {
                    "messages": messages,
                    "current_input": current_input,
                    "context": context,
                    "tools_used": tools_used,
                    "next_step": "respond"
                }
        
        # Response node
        def respond(state: AgentState) -> Dict[str, Any]:
            messages = state["messages"]
            current_input = state["current_input"]
            context = state["context"]
            tools_used = state["tools_used"]
            
            try:
                # Generate response using the prompt
                response = self.response_prompt | self.llm
                result = response.invoke({
                    "messages": messages,
                    "input": current_input,
                    "context": context
                })
                
                # Extract response text
                response_text = result.content if hasattr(result, 'content') else str(result)
                
                # Format the response text for UI display
                formatted_response = self._format_response_for_ui(response_text)
                
                # Include the full prompt in the response for UI display
                full_prompt = self.response_prompt.format(
                    messages=messages,
                    input=current_input,
                    context=context
                )
                
                # Return a valid state update
                return {
                    "messages": messages,
                    "current_input": current_input,
                    "context": context,
                    "tools_used": tools_used,
                    "final_answer": formatted_response,
                    "next_step": "end"
                }
            except Exception as e:
                print(f"Error in respond node: {str(e)}")
                return {
                    "messages": messages,
                    "current_input": current_input,
                    "context": context,
                    "tools_used": tools_used,
                    "final_answer": "I apologize, but I encountered an error while generating a response.",
                    "next_step": "end"
                }
        
        # Add nodes to graph
        graph.add_node("decide", decide)
        graph.add_node("search", search)
        graph.add_node("respond", respond)
        
        # Add edges with conditions
        graph.add_conditional_edges(
            "decide",
            lambda x: x["next_step"],
            {
                "search": "search",
                "respond": "respond"
            }
        )
        
        # Add unconditional edges
        graph.add_edge("search", "decide")
        graph.add_edge("respond", END)
        
        # Set entry point
        graph.set_entry_point("decide")
        
        return graph.compile()
    
    async def process_message(self, message: str, conversation_id: Optional[str] = None) -> dict:
        """Process a message and return the response with context."""
        try:
            print(f"ConversationMemoryAgent processing message: {message}")
            
            if not hasattr(self, 'workflow'):
                print("Error: Workflow not initialized")
                return {
                    "response": "I encountered an internal error. The conversation processing system is not properly initialized.",
                    "context_used": False,
                    "error": "Workflow not initialized",
                    "prompt": None,
                    "tools_used": []
                }
            
            # Initialize the state
            initial_state = {
                "messages": [],  # Will be populated by conversation history if needed
                "current_input": message.strip() if message else "",
                "context": [],
                "tools_used": [],
                "next_step": "decide"  # Add initial next_step
            }
            
            if not initial_state["current_input"]:
                return {
                    "response": "I received an empty message. Please provide some text to process.",
                    "context_used": False,
                    "prompt": None,
                    "tools_used": []
                }
            
            print("Invoking workflow with initial state...")
            # Use the workflow to process the message
            result = await self.workflow.ainvoke(initial_state)
            print(f"Workflow result: {result}")
            
            # Generate the full prompt for display
            try:
                full_prompt = self.response_prompt.format(
                    messages=result.get("messages", []),
                    input=result.get("current_input", ""),
                    context=result.get("context", [])
                )
            except Exception as e:
                print(f"Error generating full prompt: {str(e)}")
                full_prompt = None
            
            # Extract the response, ensuring it's a string
            response = str(result.get("final_answer", "I couldn't generate a response."))
            context = result.get("context", [])
            tools_used = result.get("tools_used", [])
            
            return {
                "response": response,
                "context_used": bool(context),
                "prompt": str(full_prompt) if full_prompt else None,
                "tools_used": tools_used,
                "conversation_id": conversation_id
            }
        except Exception as e:
            print(f"Error in ConversationMemoryAgent process_message: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "response": "I encountered an error while processing your message.",
                "context_used": False,
                "error": str(e),
                "prompt": None,
                "tools_used": []
            }
            
    async def process_conversation_upload(self, conversation_data: Dict[str, Any], stats: Optional[ProcessingStats] = None) -> Tuple[str, ProcessingStats]:
        """Process a conversation upload, storing each chat as a complete unit."""
        if stats is None:
            stats = ProcessingStats()
            
        conversation_id = conversation_data.get('id', str(uuid.uuid4()))
        print(f"Processing conversation {conversation_id}")
        skipped_details = []
        
        try:
            # Extract all messages from the chat
            messages = []
            for message in conversation_data.get('mapping', {}).values():
                if message.get('message') and isinstance(message['message'], dict):
                    content = message['message'].get('content', {})
                    text = self._extract_text_from_content(content)
                    
                    if text:
                        messages.append({
                            'text': text,
                            'role': message['message'].get('author', {}).get('role', 'user'),
                            'timestamp': datetime.fromtimestamp(
                                message['message'].get('create_time', 0)
                            ).isoformat()
                        })
            
            if messages:
                print(f"Found {len(messages)} messages in conversation {conversation_id}")
                # Update stats with message count for this conversation
                stats.start_processing_conversation(len(messages))
                
                # Process messages in chunks to avoid token limits
                for i in range(0, len(messages), MESSAGE_CHUNK_SIZE):
                    chunk = messages[i:i + MESSAGE_CHUNK_SIZE]
                    print(f"Processing chunk {i//MESSAGE_CHUNK_SIZE + 1} of conversation {conversation_id}")
                    
                    # Create embedding for the chunk
                    chat_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chunk])
                    
                    # Check token count and chunk if necessary
                    if count_tokens(chat_text) > MAX_TOKENS:
                        text_chunks = chunk_text(chat_text)
                        print(f"Splitting chunk into {len(text_chunks)} text chunks")
                        for idx, text_chunk in enumerate(text_chunks):
                            chunk_embedding = await self.embeddings_manager.get_embedding_async(text_chunk)
                            chunk_id = f"{conversation_id}_{i}_{idx}"
                            print(f"Storing text chunk {idx + 1} as {chunk_id}")
                            
                            # Store the chunk
                            await self.vector_store.add_vector_async(
                                key=f"chat:{chunk_id}",
                                vector=chunk_embedding,
                                metadata={
                                    'conversation_id': conversation_id,
                                    'chunk_id': chunk_id,
                                    'messages': chunk,
                                    'text': text_chunk,
                                    'timestamp': chunk[0]['timestamp'],
                                    'last_used': None,
                                    'is_chunk': True,
                                    'total_chunks': len(text_chunks)
                                }
                            )
                            # Update progress after each chunk
                            stats.update_progress()
                            print(f"Progress: {stats.chunks_processed}/{stats.total_chunks} chunks")
                    else:
                        # Store the chunk as is
                        chunk_embedding = await self.embeddings_manager.get_embedding_async(chat_text)
                        chunk_id = f"{conversation_id}_{i}"
                        print(f"Storing chunk as {chunk_id}")
                        
                        await self.vector_store.add_vector_async(
                            key=f"chat:{chunk_id}",
                            vector=chunk_embedding,
                            metadata={
                                'conversation_id': conversation_id,
                                'chunk_id': chunk_id,
                                'messages': chunk,
                                'text': chat_text,
                                'timestamp': chunk[0]['timestamp'],
                                'last_used': None,
                                'is_chunk': False
                            }
                        )
                        # Update progress after each chunk
                        stats.update_progress()
                        print(f"Progress: {stats.chunks_processed}/{stats.total_chunks} chunks")
                
                stats.update(processed=1)
                print(f"Completed processing conversation {conversation_id}")
            else:
                print(f"No messages found in conversation {conversation_id}")
                stats.update(skipped=1)
                skipped_details.append({
                    "conversation_id": conversation_id,
                    "reason": "No messages found"
                })
                
        except Exception as e:
            print(f"Error processing conversation {conversation_id}: {str(e)}")
            stats.update(errors=1)
            raise
        
        return conversation_id, stats
    
    def _extract_text_from_content(self, content: Any) -> str:
        """Extract text from various content formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            if "parts" in content:
                parts = content["parts"]
                if isinstance(parts, list):
                    return " ".join(str(part) for part in parts if isinstance(part, (str, int, float)))
            elif "text" in content:
                return content["text"]
        return ""
    
    def _format_conversation_snippet(self, text: str, timestamp: str = "") -> str:
        """Format a conversation snippet with proper labels, formatting, and timestamp."""
        # Add timestamp header if provided
        formatted_lines = []
        if timestamp:
            try:
                # Convert timestamp to readable format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_lines.append(f"[Conversation from {dt.strftime('%B %d, %Y at %I:%M %p')}]")
                formatted_lines.append("")  # Add blank line after timestamp
            except:
                formatted_lines.append(f"[Conversation from {timestamp}]")
                formatted_lines.append("")  # Add blank line after timestamp
        
        # Replace labels and split into lines
        lines = text.replace("Assistant:", "You:").replace("User:", "Me:").split('\n')
        current_message = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Me:") or line.startswith("You:"):
                # If we have a previous message, add it
                if current_message:
                    formatted_lines.append(" ".join(current_message))
                    formatted_lines.append("")  # Add blank line between messages
                current_message = [line]
            else:
                current_message.append(line)
        
        # Add the last message if exists
        if current_message:
            formatted_lines.append(" ".join(current_message))
        
        return "\n".join(formatted_lines)

    def _extract_relevant_clip(self, text: str, query: str) -> str:
        """Extract the most relevant part of a conversation based on the query."""
        # Split into message pairs
        messages = []
        current_pair = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(("User:", "Assistant:", "Me:", "You:")):
                if current_pair and len(current_pair) == 2:
                    messages.append("\n".join(current_pair))
                    current_pair = []
                current_pair.append(line)
            elif current_pair:
                current_pair[-1] += " " + line
        
        if current_pair:
            messages.append("\n".join(current_pair))
        
        # If only one message pair, return it formatted
        if len(messages) <= 1:
            return text
        
        # Find most relevant message pair
        most_relevant = None
        highest_similarity = -1
        
        for msg_pair in messages:
            similarity = self.embeddings_manager.compute_similarity(query, msg_pair)
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_relevant = msg_pair
        
        return most_relevant if most_relevant else text

    def _search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search for relevant conversations using the query."""
        try:
            if not isinstance(query, str):
                print(f"Invalid query type: {type(query)}")
                return []
                
            # Clean the query
            query = query.strip()
            if not query:
                print("Empty query after cleaning")
                return []
            
            # Check if this is a request for a specific conversation
            is_specific_request = any(phrase in query.lower() for phrase in [
                "show me", "that conversation", "the conversation", "snippet", "what did we say",
                "remember when", "you said", "i said", "we talked about"
            ])
            
            # If asking for a specific conversation and we have a last mentioned one, use it
            if is_specific_request and self.last_mentioned_conversation:
                print(f"Using last mentioned conversation: {self.last_mentioned_conversation}")
                # Extract relevant clip and format with timestamp
                text = self._extract_relevant_clip(
                    self.last_mentioned_conversation['text'],
                    query
                )
                self.last_mentioned_conversation['text'] = self._format_conversation_snippet(
                    text,
                    self.last_mentioned_conversation.get('timestamp', '')
                )
                return [self.last_mentioned_conversation]
            
            # Get embedding for the query
            query_embedding = self.embeddings_manager.get_embedding(query)
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Search with appropriate parameters
            if is_specific_request:
                # Use higher similarity threshold and fewer results for specific requests
                results = self.vector_store.search_similar(
                    query_vector=query_embedding,
                    top_k=1,
                    min_similarity=0.85
                )
            else:
                # Use lower threshold and more results for general context
                results = self.vector_store.search_similar(
                    query_vector=query_embedding,
                    top_k=3,
                    min_similarity=0.7
                )
            
            if not results:
                print("No similar conversations found")
                return []
            
            # Format results
            formatted_results = []
            for result in results:
                if isinstance(result, dict) and 'text' in result:
                    # Extract relevant clip and format with timestamp
                    text = self._extract_relevant_clip(result['text'], query) if is_specific_request else result['text']
                    formatted_text = self._format_conversation_snippet(
                        text,
                        result.get('timestamp', '')
                    )
                    
                    formatted_result = {
                        'text': formatted_text,
                        'similarity': result.get('similarity', 0),
                        'conversation_id': result.get('conversation_id', 'unknown'),
                        'timestamp': result.get('timestamp', '')
                    }
                    formatted_results.append(formatted_result)
                    
                    # Store the most relevant result
                    if not is_specific_request and len(formatted_results) == 1:
                        self.last_mentioned_conversation = formatted_result
            
            # Sort by similarity for specific requests
            if is_specific_request:
                formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
                formatted_results = formatted_results[:1]
            
            print(f"Found {len(formatted_results)} relevant conversations")
            return formatted_results
            
        except Exception as e:
            print(f"Error in _search_conversations: {str(e)}")
            return []
        
    def _rerank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results considering active memories and relevance."""
        scored_results = []
        current_time = datetime.now()
        
        for result in results:
            score = result.get('similarity', 0)
            memory_id = f"{result['conversation_id']}:{result.get('chunk_id', '')}"
            
            # Boost score if memory is active
            if memory_id in self.active_memories:
                time_since_used = (current_time - self.active_memories[memory_id]["last_used"]).total_seconds()
                recency_boost = max(0, 1 - (time_since_used / 3600))  # Boost decreases over time
                score += recency_boost * 0.5
            
            scored_results.append((score, result))
        
        # Sort by score and return results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for _, result in scored_results]
        
    def _update_active_memory(self, memory: Dict[str, Any]):
        """Update the active memories with a recently used memory."""
        memory_id = f"{memory['conversation_id']}:{memory.get('chunk_id', '')}"
        self.active_memories[memory_id] = {
            "memory": memory,
            "last_used": datetime.now()
        }
    
    def _get_active_memories(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve active memories that might be relevant to the query."""
        current_time = datetime.now()
        active_memories = []
        
        for memory_id, memory_data in self.active_memories.items():
            # Check if memory is still active (within last hour)
            if (current_time - memory_data["last_used"]).total_seconds() < 3600:
                active_memories.append(memory_data["memory"])
        
        return active_memories

    def _format_response_for_ui(self, text: str) -> str:
        """Format the response text for proper display in the UI."""
        # Split into paragraphs (split on double newlines first)
        paragraphs = [p.strip() for p in text.split('\n\n')]
        if len(paragraphs) == 1:  # If no double newlines, try single newlines
            paragraphs = [p.strip() for p in text.split('\n')]
        
        formatted_lines = []
        in_conversation = False
        
        for paragraph in paragraphs:
            if not paragraph:
                continue
            
            # Handle conversation headers
            if paragraph.startswith('[Conversation from'):
                formatted_lines.extend(['', paragraph, ''])
                in_conversation = True
                continue
            
            # Handle messages in conversation
            if paragraph.startswith(('Me:', 'You:')):
                formatted_lines.extend(['', paragraph])
                continue
            
            # Handle regular paragraphs
            if in_conversation and formatted_lines and formatted_lines[-1].startswith(('Me:', 'You:')):
                # Continue previous message
                formatted_lines[-1] += " " + paragraph
            else:
                # New paragraph - add blank line before if needed
                if formatted_lines and not formatted_lines[-1] == '':
                    formatted_lines.append('')
                formatted_lines.append(paragraph)
        
        # Join with newlines and ensure proper spacing
        formatted_text = '\n'.join(line for line in formatted_lines if line is not None)
        
        # Ensure proper paragraph breaks in the main text
        # Replace any sequence of 3 or more newlines with just 2
        while '\n\n\n' in formatted_text:
            formatted_text = formatted_text.replace('\n\n\n', '\n\n')
        
        return formatted_text

class ChatProcessor:
    def __init__(self, api_key: str, vector_store: VectorStore, embeddings_manager: EmbeddingsManager):
        """Initialize the chat processor with required components."""
        self.api_key = api_key
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.memory_agent = ConversationMemoryAgent(vector_store, embeddings_manager, api_key)
        self._progress_queue = asyncio.Queue()
        
    async def get_progress_stream(self):
        """Stream progress updates."""
        try:
            while True:
                data = await self._progress_queue.get()
                if data.get("status") == "complete":
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.1)  # Small delay to prevent overwhelming the client
        except asyncio.CancelledError:
            # Handle client disconnect
            pass
            
    async def process_conversations(self, conversations: List[Dict[str, Any]], stats: ProcessingStats):
        """Process a list of conversations in the background."""
        results = []
        total = len(conversations)
        skipped_details = []
        
        for i, conversation in enumerate(conversations):
            try:
                # Process the conversation
                conversation_id = conversation.get('id', 'unknown')
                conversation_id, updated_stats = await self.memory_agent.process_conversation_upload(conversation, stats)
                results.append({"conversation_id": conversation_id})
                
                # Send progress update
                await self._progress_queue.put({
                    "status": "processing",
                    "processed": i + 1,
                    "total": total,
                    "success_count": stats.processed,
                    "skipped_count": stats.skipped,
                    "error_count": stats.errors,
                    "chunks_processed": stats.chunks_processed,
                    "total_chunks": stats.total_chunks,
                    "skipped_details": skipped_details
                })
                
            except Exception as e:
                print(f"Error processing conversation {i}: {str(e)}")
                stats.update(errors=1)
                
        # Get final stats
        memory_stats = await self.vector_store.get_memory_stats()
        
        # Send completion message
        await self._progress_queue.put({
            "status": "complete",
            "message": f"Successfully processed {stats.processed} conversations",
            "results": results,
            "stats": stats.to_dict(),
            "memory_stats": memory_stats,
            "skipped_details": skipped_details
        })
        
    async def process_message(self, message: str, conversation_id: Optional[str] = None) -> dict:
        """Process a message and return the response with context."""
        try:
            print(f"Processing message via memory agent: {message}")
            return await self.memory_agent.process_message(message, conversation_id)
        except Exception as e:
            print(f"Error in chat processor process_message: {str(e)}")
            return {
                "response": "I encountered an error while processing your message.",
                "context_used": False,
                "error": str(e),
                "prompt": None,
                "tools_used": []
            }
