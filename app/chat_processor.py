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
        """Initialize the conversation memory agent with enhanced memory capabilities."""
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.api_key = api_key
        self.chat_history = ChatMessageHistory()
        self.active_memories = {}
        self.last_mentioned_conversations = []
        self.last_response_context = None
        self.user_model = {}
        
        # Create LLMs with different creativity levels
        self.analysis_llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_key,
            max_tokens=1000,
            temperature=0,  # Keep analysis deterministic
            model_kwargs={
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        )
        
        # Synthesis LLM with moderate creativity
        synthesis_kwargs = {
            "top_p": 0.9,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.2
        }
        self.synthesis_llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_key,
            max_tokens=1500,
            temperature=0.3,
            model_kwargs=synthesis_kwargs
        )
        
        # Response LLM with high creativity
        response_kwargs = {
            "top_p": 0.9,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.4
        }
        self.response_llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=self.api_key,
            max_tokens=2000,
            temperature=0.7,
            model_kwargs=response_kwargs
        )
        
        # Create enhanced prompts
        print("\n=== Template Creation Debug ===")
        print("Creating query analysis prompt...")
        
        # Debug the system message
        query_system_message = (
            """You are an expert at analyzing questions to determine what context would be needed to provide a complete and accurate answer.

For any given query, determine what information would be needed to provide a complete answer.

Your task is to analyze this query: {query}

You must respond with a JSON object using exactly this structure:
{{
    "search_strategy": {{
        "primary_aspects": [],      
        "semantic_variations": [],  
        "temporal_scope": "",      
        "context_type": ""         
    }},
    "required_context": {{
        "interests": false,         
        "preferences": false,       
        "behaviors": false,         
        "opinions": false,          
        "knowledge": false,         
        "emotions": false          
    }}
}}""")

        print(f"Query system message (first 100 chars): {query_system_message[:100]}")
        print(f"Query system message length: {len(query_system_message)}")
        
        # Create and debug the query template
        self.query_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing questions to determine what context would be needed to provide a complete and accurate answer.

For any given query, determine what information would be needed to provide a complete answer.

Your task is to analyze this query: {query}

You must respond with a JSON object using exactly this structure:
{{
    "search_strategy": {{
        "primary_aspects": [],      
        "semantic_variations": [],  
        "temporal_scope": "",      
        "context_type": ""         
    }},
    "required_context": {{
        "interests": false,         
        "preferences": false,       
        "behaviors": false,         
        "opinions": false,          
        "knowledge": false,         
        "emotions": false          
    }}
}}"""),
            ("human", "Please analyze the query and provide your JSON response.")
        ])
        
        print(f"Query template input variables: {self.query_analysis_prompt.input_variables}")
        print(f"Query template message types: {[type(m) for m in self.query_analysis_prompt.messages]}")
        
        # Debug the memory synthesis template
        print("\nCreating memory synthesis prompt...")
        self.memory_synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the provided memories and synthesize insights. Be creative in finding patterns and connections.
Focus on:
1. Identifying unique patterns and unexpected trends
2. Extracting subtle behavioral traits
3. Noting emotional patterns and their evolution
4. Finding interesting connections across time
5. Drawing novel insights while maintaining accuracy

Format your response as JSON with these fields:
{{
    "interests": [],        
    "preferences": [],      
    "behavioral_traits": [], 
    "emotional_patterns": [], 
    "recurring_themes": [], 
    "key_insights": []     
}}

Base your insights ONLY on the provided memories. DO NOT make assumptions or invent details.

Memories to analyze: {memories}"""),
            ("human", "Please provide your analysis.")
        ])
        
        print(f"Memory template input variables: {self.memory_synthesis_prompt.input_variables}")
        print(f"Memory template message types: {[type(m) for m in self.memory_synthesis_prompt.messages]}")
        
        self.response_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                """You are a thoughtful AI assistant having a conversation. Your responses should be:
1. STRICTLY based on actual retrieved memories when discussing past interactions
2. Natural and conversational for general topics
3. Helpful and specific even when no memories are available

CRITICAL RULES:
1. NEVER invent or fabricate memories or past interactions
2. When memories are found: 
    a. Use them to provide personalized insights and specific suggestions
    b. Draw connections between different interests/preferences found
    c. Consider both explicit statements ("I love X") and implicit preferences
3. When no memories are found: 
    a. For personal queries: Ask relevant follow-up questions to gather information
    b. For general queries: Provide thoughtful, general advice while gathering more context
4. Always interpret queries from the USER's perspective first
5. Keep conversation/document IDs in your working memory but DO NOT reference them in responses

For gift/preference queries:
1. First use any found memories to identify:
    a. Explicitly stated interests/hobbies
    b. Positive reactions to topics/activities
    c. Mentioned wishes or wants
    d. Past discussions about preferences
2. Then synthesize these into specific gift suggestions
3. If suggesting something, explain why based on their interests
4. Only ask follow-up questions if no relevant memories are found

Example responses:

With memories:
"Based on your interests, I think you'd particularly enjoy [specific suggestion based on actual memories]. You've shown enthusiasm for [observed interest], so perhaps [related suggestion]. Given your interest in [another observed topic], you might also appreciate [another specific suggestion]."

Without memories:
"To suggest gifts that would really suit you, could you tell me more about:
1. What kinds of activities do you enjoy most?
2. Are there any hobbies or topics you're particularly passionate about?
3. Do you prefer practical items or experiences?"

Remember: 
1. Stay focused on helping the user
2. Be specific when you have information
3. Ask good questions when you need more context
4. Never pretend to have information you don't have"""
            )),
            ("human", (
                """Query: {input}
Analysis: {analysis}
Retrieved Memories: {memories}
Synthesis: {synthesis}"""
            ))
        ])
        
        # Add message-specific prompt storage
        self.message_prompts = {}  # Store prompts by message ID

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine search strategy and required context."""
        try:
            print("\n=== Query Analysis Debug ===")
            print(f"Input query: {query}")
            print(f"Creating chain with prompt type: {type(self.query_analysis_prompt)}")
            
            # Create chain
            chain = self.query_analysis_prompt | self.analysis_llm
            print("Chain created successfully")
            
            # Debug input preparation
            input_dict = {"query": query}
            print(f"Input dictionary: {input_dict}")
            print(f"Expected variables: {self.query_analysis_prompt.input_variables}")
            print(f"Provided variables: {list(input_dict.keys())}")
            
            # Execute chain
            print("Executing chain...")
            result = await chain.ainvoke(input_dict)
            print(f"Chain execution complete. Result type: {type(result)}")
            
            # Extract content
            if hasattr(result, 'content'):
                content = result.content
                print("Extracted content from AIMessage")
            else:
                content = str(result)
                print("Converted result to string")
            print(f"Content: {content}")
            
            # Parse JSON
            try:
                print("Attempting to parse JSON...")
                analysis = json.loads(content)
                print("JSON parsing successful")
                print(f"Analysis structure: {list(analysis.keys())}")
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {str(e)}")
                print(f"Failed content: {content}")
                # Use fallback analysis
                analysis = {
                    "search_strategy": {
                        "primary_aspects": ["interests", "hobbies", "preferences", "wishlist"],
                        "semantic_variations": [
                            "I love", "I enjoy", "I like", "I want",
                            "my favorite", "I wish", "I need", "I've been wanting"
                        ],
                        "temporal_scope": "all",
                        "context_type": "synthesis"
                    },
                    "required_context": {
                        "interests": True,
                        "preferences": True,
                        "behaviors": True,
                        "opinions": True,
                        "knowledge": True,
                        "emotions": True
                    }
                }
                print("Using fallback analysis")
            
            print(f"Final analysis: {json.dumps(analysis, indent=2)}")
            return analysis
            
        except Exception as e:
            print(f"Error in query analysis: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            print("Using error fallback analysis")
            return {
                "search_strategy": {
                    "primary_aspects": ["interests", "preferences"],
                    "semantic_variations": ["I enjoy", "I like", "I want"],
                    "temporal_scope": "all",
                    "context_type": "synthesis"
                },
                "required_context": {
                    "interests": True,
                    "preferences": True,
                    "behaviors": True,
                    "opinions": False,
                    "knowledge": False,
                    "emotions": True
                }
            }

    async def _synthesize_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize insights from multiple memories."""
        try:
            if not memories:
                print("No memories to synthesize")
                return {}
                
            print("\n=== Memory Synthesis Debug ===")
            print(f"Number of memories to process: {len(memories)}")
            
            # Format memories for synthesis
            formatted_memories = []
            for memory in memories:
                formatted_memories.append({
                    'text': memory.get('text', ''),
                    'timestamp': memory.get('timestamp', ''),
                    'similarity': memory.get('similarity', 0)
                })
            
            print(f"First memory sample: {json.dumps(formatted_memories[0] if formatted_memories else {}, indent=2)}")
            print(f"Memory synthesis prompt variables: {self.memory_synthesis_prompt.input_variables}")
            
            # Debug input preparation
            input_dict = {"memories": json.dumps(formatted_memories, ensure_ascii=False)}
            print(f"Input dictionary keys: {list(input_dict.keys())}")
            print(f"Expected variables: {self.memory_synthesis_prompt.input_variables}")
            
            # Create and execute chain
            print("Creating synthesis chain...")
            chain = self.memory_synthesis_prompt | self.synthesis_llm
            print("Executing synthesis chain...")
            result = await chain.ainvoke(input_dict)
            print(f"Synthesis result type: {type(result)}")
            
            # Parse the response
            synthesis_text = result.content if hasattr(result, 'content') else str(result)
            print(f"Raw synthesis text (first 100 chars): {synthesis_text[:100]}")
            
            try:
                print("Attempting to parse synthesis JSON...")
                synthesis = json.loads(synthesis_text)
                print(f"Synthesis structure: {list(synthesis.keys())}")
                return synthesis
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {str(e)}")
                print("Falling back to text parsing")
                synthesis = {
                    'interests': [],
                    'preferences': [],
                    'behavioral_traits': [],
                    'emotional_patterns': [],
                    'recurring_themes': [],
                    'key_insights': []
                }
                return synthesis
            
        except Exception as e:
            print(f"Error in memory synthesis: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            return {}

    def _search_conversations(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute semantic search based on query analysis."""
        try:
            print(f"Executing semantic search for: {query}")
            all_results = []
            search_strategy = analysis.get('search_strategy', {})
            required_context = analysis.get('required_context', {})
            
            # Primary search based on main query
            query_embedding = self.embeddings_manager.get_embedding(query)
            if query_embedding:
                base_results = self.vector_store.search_similar(
                    query_vector=query_embedding,
                    top_k=5,
                    min_similarity=0.6  # Lower threshold for initial search
                )
                if base_results:
                    verified_base = self._verify_and_format_results(base_results)
                    all_results.extend(verified_base)
            
            # Search for each primary aspect with query context
            for aspect in search_strategy.get('primary_aspects', []):
                # Create contextual queries that combine the original query with the aspect
                contextual_queries = [
                    f"When discussing {query}, mentioned {aspect}",
                    f"In context of {query}, expressed {aspect}",
                    f"While talking about {query}, showed {aspect}"
                ]
                
                for aspect_query in contextual_queries:
                    aspect_embedding = self.embeddings_manager.get_embedding(aspect_query)
                    if aspect_embedding:
                        aspect_results = self.vector_store.search_similar(
                            query_vector=aspect_embedding,
                            top_k=3,
                            min_similarity=0.6
                        )
                        if aspect_results:
                            verified_aspect = self._verify_and_format_results(aspect_results)
                            all_results.extend(verified_aspect)
            
            # Search semantic variations with context preservation
            for variation in search_strategy.get('semantic_variations', []):
                # Create variations that maintain the original query context
                contextual_variations = [
                    variation,
                    f"{variation} that could be a gift",
                    f"{variation} as a present",
                    f"{variation} to receive"
                ]
                
                for var_query in contextual_variations:
                    variation_embedding = self.embeddings_manager.get_embedding(var_query)
                    if variation_embedding:
                        variation_results = self.vector_store.search_similar(
                            query_vector=variation_embedding,
                            top_k=3,
                            min_similarity=0.55  # Lower threshold for variations
                        )
                        if variation_results:
                            verified_variation = self._verify_and_format_results(variation_results)
                            all_results.extend(verified_variation)
            
            # Additional context searches based on required_context
            if any(required_context.values()):
                context_queries = []
                
                # Build rich context queries
                if required_context.get('interests'):
                    context_queries.extend([
                        "things I'm passionate about",
                        "activities I enjoy",
                        "what excites me most"
                    ])
                if required_context.get('preferences'):
                    context_queries.extend([
                        "things I prefer",
                        "what I look for in",
                        "qualities I appreciate"
                    ])
                if required_context.get('behaviors'):
                    context_queries.extend([
                        "how I spend my time",
                        "what I do often",
                        "my typical activities"
                    ])
                if required_context.get('emotions'):
                    context_queries.extend([
                        "things that make me happy",
                        "what I get excited about",
                        "experiences I value"
                    ])
                
                for context_query in context_queries:
                    context_embedding = self.embeddings_manager.get_embedding(context_query)
                    if context_embedding:
                        context_results = self.vector_store.search_similar(
                            query_vector=context_embedding,
                            top_k=3,
                            min_similarity=0.55
                        )
                        if context_results:
                            verified_context = self._verify_and_format_results(context_results)
                            all_results.extend(verified_context)
            
            # Deduplicate and sort results
            unique_results = self._deduplicate_results(all_results)
            
            # Sort by similarity and limit results
            unique_results.sort(key=lambda x: x['similarity'], reverse=True)
            final_results = unique_results[:10]  # Increased limit for comprehensive context
            
            print(f"Found {len(final_results)} verified conversations")
            return final_results
            
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

    def _verify_and_format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify and format search results."""
        verified_results = []
        seen_conversations = set()
        
        for result in results:
            if not isinstance(result, dict) or 'text' not in result:
                continue
                
            conversation_id = result.get('conversation_id', 'unknown')
            if conversation_id in seen_conversations:
                continue
                
            if not self._verify_conversation_participants(result['text']):
                continue
                
            formatted_result = {
                'text': self._format_conversation_snippet(
                    result['text'],
                    result.get('timestamp', '')
                ),
                'similarity': result.get('similarity', 0),
                'conversation_id': conversation_id,
                'timestamp': result.get('timestamp', ''),
                'is_direct_interaction': True,
                'metadata': result.get('metadata', {})
            }
            
            verified_results.append(formatted_result)
            seen_conversations.add(conversation_id)
            
        return verified_results

    async def process_message(self, message: str, conversation_id: Optional[str] = None) -> dict:
        """Process a message using enhanced multi-stage reasoning."""
        try:
            print(f"Processing message with enhanced reasoning: {message}")
            message_id = str(uuid.uuid4())
            
            # Stage 1: Query Analysis
            query_analysis = await self._analyze_query(message)
            print(f"Query analysis: {query_analysis}")
            
            # Initialize search result variables
            direct_results = []
            pattern_results = []
            unique_results = []
            
            # Stage 2: Memory Retrieval
            search_results = self._search_conversations(message, query_analysis)
            if search_results:
                direct_results = search_results
                unique_results = self._deduplicate_results(search_results)
            
            # Stage 3: Memory Synthesis
            synthesis = {}
            if unique_results:
                synthesis = await self._synthesize_memories(unique_results)
            
            # Stage 4: Response Generation
            response = self.response_generation_prompt | self.response_llm
            result = await response.ainvoke({
                "input": message,
                "analysis": json.dumps(query_analysis, ensure_ascii=False),
                "memories": json.dumps(unique_results, ensure_ascii=False) if unique_results else "No relevant memories found",
                "synthesis": json.dumps(synthesis, ensure_ascii=False)
            })
            
            # Extract and format response
            response_text = result.content if hasattr(result, 'content') else str(result)
            formatted_response = self._format_response_for_ui(response_text)
            
            # Format prompt details for UI
            prompt_details = {
                "Query Analysis": query_analysis,
                "Memory Search": {
                    "Direct Results": len(direct_results),
                    "Pattern Results": len(pattern_results),
                    "Total Unique Results": len(unique_results),
                    "Retrieved Memories": [
                        {
                            "text": result["text"],
                            "timestamp": result.get("timestamp", ""),
                            "similarity": result.get("similarity", 0)
                        }
                        for result in unique_results
                    ]
                },
                "Memory Synthesis": synthesis if synthesis else "No synthesis performed",
                "Response Generation": {
                    "Input": message,
                    "Analysis": query_analysis,
                    "Context Used": bool(unique_results)
                }
            }
            
            # Store prompt details for this message
            self.message_prompts[message_id] = prompt_details
            
            return {
                "response": formatted_response,
                "context_used": bool(unique_results),
                "synthesis_performed": bool(synthesis),
                "query_type": query_analysis.get('Query Type', 'unknown'),
                "conversation_id": conversation_id,
                "message_id": message_id,
                "prompt": json.dumps(prompt_details, indent=2, ensure_ascii=False)
            }
            
        except Exception as e:
            print(f"Error in enhanced process_message: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "response": "I encountered an error while processing your message.",
                "context_used": False,
                "error": str(e),
                "conversation_id": conversation_id,
                "message_id": str(uuid.uuid4()),
                "prompt": None
            }
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate search results while preserving order and relevance."""
        seen = set()
        unique_results = []
        
        for result in results:
            conversation_id = result.get('conversation_id', 'unknown')
            if conversation_id not in seen:
                seen.add(conversation_id)
                unique_results.append(result)
        
        return unique_results
    
    def _update_user_model(self, analysis: Dict[str, Any], synthesis: Dict[str, Any]):
        """Update the user model with new insights."""
        try:
            # Update interaction patterns
            if 'query_type' in analysis:
                self.user_model.setdefault('interaction_patterns', {})
                query_type = analysis['query_type']
                self.user_model['interaction_patterns'][query_type] = self.user_model['interaction_patterns'].get(query_type, 0) + 1
            
            # Update behavioral traits
            if 'behavioral_traits' in synthesis:
                self.user_model.setdefault('behavioral_traits', {})
                for trait in synthesis['behavioral_traits']:
                    self.user_model['behavioral_traits'][trait] = self.user_model['behavioral_traits'].get(trait, 0) + 1
            
            # Update areas of interest
            if 'interests' in synthesis:
                self.user_model.setdefault('interests', {})
                for interest in synthesis['interests']:
                    self.user_model['interests'][interest] = self.user_model['interests'].get(interest, 0) + 1
            
            # Update emotional patterns
            if 'emotional_patterns' in synthesis:
                self.user_model.setdefault('emotional_patterns', [])
                self.user_model['emotional_patterns'].extend(synthesis.get('emotional_patterns', []))
            
            # Trim history if needed
            if len(self.user_model.get('emotional_patterns', [])) > 100:
                self.user_model['emotional_patterns'] = self.user_model['emotional_patterns'][-100:]
            
        except Exception as e:
            print(f"Error updating user model: {str(e)}")

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
        # Note: From AI's perspective, "Me" is the assistant and "You" is the user
        lines = text.replace("assistant:", "Me:").replace("user:", "You:").split('\n')
        current_speaker = None
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
                current_speaker = line[:3]  # "Me:" or "You:"
                current_message = [line]
            else:
                # If no speaker prefix and we have a current speaker, append to current message
                if current_speaker:
                    current_message.append(line)
                else:
                    # If no current speaker, treat as a new message with default speaker
                    if current_message:
                        formatted_lines.append(" ".join(current_message))
                        formatted_lines.append("")
                    current_message = [line]
        
        # Add the last message if exists
        if current_message:
            formatted_lines.append(" ".join(current_message))
        
        # Ensure proper spacing
        return "\n".join(formatted_lines).strip()

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

    def _verify_conversation_participants(self, text: str) -> bool:
        """Verify that a conversation is between the AI assistant and the user."""
        # Look for patterns indicating AI-user interaction
        has_user = bool(re.search(r'\b(user:|User:|you:|You:)', text))
        has_assistant = bool(re.search(r'\b(assistant:|Assistant:|me:|Me:)', text))
        
        # Check for third-party mentions or other participant indicators
        third_party_indicators = [
            'friend', 'they', 'them', 'he said', 'she said', 'they said',
            'someone', 'people', 'others', 'we played', 'I played with'
        ]
        
        has_third_party = any(indicator in text.lower() for indicator in third_party_indicators)
        
        # Return True only if we have both user and assistant messages and no third-party indicators
        return has_user and has_assistant and not has_third_party

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
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        formatted_lines = []
        in_conversation = False
        current_speaker = None
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Handle conversation references (preserve conversation_id)
            if '(conversation_id:' in paragraph:
                # Ensure the conversation_id appears on its own line
                parts = paragraph.split('(conversation_id:', 1)
                if len(parts) == 2:
                    id_part = parts[1].split(')', 1)
                    if len(id_part) == 2:
                        formatted_lines.extend([
                            parts[0].strip(),
                            f"(conversation_id:{id_part[0]})",
                            id_part[1].strip() if id_part[1].strip() else ""
                        ])
                        continue
            
            # Handle conversation headers
            if paragraph.startswith('[Conversation from'):
                if formatted_lines and formatted_lines[-1]:
                    formatted_lines.append('')  # Add blank line before header
                formatted_lines.append(paragraph)
                formatted_lines.append('')  # Add blank line after header
                in_conversation = True
                current_speaker = None
                continue
            
            # Handle messages in conversation
            if paragraph.startswith(('Me:', 'You:')):
                if current_speaker:
                    formatted_lines.append('')  # Add blank line between messages
                formatted_lines.append(paragraph)
                current_speaker = paragraph[:3]
                continue
            
            # Handle regular paragraphs
            if in_conversation and current_speaker:
                # Continue previous message
                formatted_lines[-1] += " " + paragraph
            else:
                # New paragraph
                if formatted_lines and formatted_lines[-1]:
                    formatted_lines.append('')  # Add blank line between paragraphs
                formatted_lines.append(paragraph)
                in_conversation = False
                current_speaker = None
        
        # Join with newlines and clean up any excessive spacing
        formatted_text = '\n'.join(formatted_lines)
        
        # Clean up excessive newlines while preserving intentional spacing
        while '\n\n\n' in formatted_text:
            formatted_text = formatted_text.replace('\n\n\n', '\n\n')
        
        return formatted_text.strip()

class ChatProcessor:
    def __init__(self, vector_store: VectorStore, embeddings_manager: EmbeddingsManager, api_key: str):
        """Initialize the chat processor."""
        if not isinstance(vector_store, VectorStore):
            raise ValueError("vector_store must be an instance of VectorStore")
        if not isinstance(embeddings_manager, EmbeddingsManager):
            raise ValueError("embeddings_manager must be an instance of EmbeddingsManager")
        if not isinstance(api_key, str):
            raise ValueError("api_key must be a string")
        if not api_key.strip():
            raise ValueError("api_key cannot be empty")
            
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.api_key = api_key
        self._progress_queue = asyncio.Queue()
        
        # Initialize the memory agent
        print(f"Initializing ConversationMemoryAgent with api_key type: {type(api_key)}")
        self.memory_agent = ConversationMemoryAgent(
            vector_store=vector_store,
            embeddings_manager=embeddings_manager,
            api_key=api_key
        )
    
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
            # Only log once at the start
            print(f"Processing chat request: {message}")
            
            # Delegate to memory agent for processing
            result = await self.memory_agent.process_message(message, conversation_id)
            
            # Return the result without additional processing
            return result
            
        except Exception as e:
            print(f"Error in chat processor process_message: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "response": "I encountered an error while processing your message.",
                "context_used": False,
                "error": str(e),
                "conversation_id": conversation_id
            }
