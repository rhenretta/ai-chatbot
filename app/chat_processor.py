from typing import List, Dict, Any, Optional, Tuple, Union, TypedDict, Annotated
from datetime import datetime, timedelta
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
        
        # Enhanced tone tracking
        self.conversation_tone = {
            "conversation_style": "casual_chat",  # Default to casual
            "emotional_context": "friendly",
            "response_approach": "chat_naturally",
            "formality_level": "casual"
        }
        self.tone_history = []  # Track recent tones to detect patterns
        self.tone_transition_threshold = 0.7  # Threshold for tone changes
        
        # Tone transition weights (how much each aspect influences tone changes)
        self.tone_weights = {
            "conversation_style": 0.3,
            "emotional_context": 0.3,
            "response_approach": 0.2,
            "formality_level": 0.2
        }
        
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

Consider the following aspects:
1. Query Style and Intent:
   - Is this a casual chat or formal inquiry?
   - Is the user reminiscing or seeking specific information?
   - Are they looking for facts or more of a friendly discussion?
   - What's the underlying social context of their question?

2. Time Context:
   - Is there a specific time period mentioned?
   - Are they asking about a holiday or special event?
   - Is it about a recent or past time period?
   - How precise should the time filtering be?

3. Emotional Context:
   - What's the mood/vibe of the question?
   - Is this a light conversation starter or serious inquiry?
   - Are they sharing or seeking a moment of connection?
   - What emotional response would feel natural?

4. Response Approach:
   - Should we chat like friends or provide formal information?
   - Do they want precise details or more of a general reflection?
   - Would they prefer highlights and impressions over exact quotes?
   - How can we make this feel like a natural conversation?

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
    }},
    "tone_analysis": {{
        "conversation_style": "",    
        "emotional_context": "",     
        "response_approach": "",     
        "formality_level": ""       
    }}
}}""")

        print(f"Query system message (first 100 chars): {query_system_message[:100]}")
        print(f"Query system message length: {len(query_system_message)}")
        
        # Create query analysis prompt with proper structure
        self.query_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing questions to determine what context would be needed to provide a complete and accurate answer.

For any given query, determine what information would be needed to provide a complete answer.

Your task is to analyze this query: {query}

Consider the following aspects:
1. Query Style and Intent
2. Time Context
3. Emotional Context
4. Response Approach

Respond with a JSON object using this structure:
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
    }},
    "tone_analysis": {{
        "conversation_style": "",    
        "emotional_context": "",     
        "response_approach": "",     
        "formality_level": ""       
    }}
}}"""),
            ("human", "{query}")
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

Pay special attention to:
1. Emotional Context:
   - Overall emotional tone of conversations
   - Changes in emotional state across interactions
   - Topics that evoke strong emotions
   - Patterns in emotional responses

2. Conversation Style:
   - Level of formality in interactions
   - Use of humor or playfulness
   - Depth of philosophical/personal discussions
   - Preferred conversation topics

3. Relationship Dynamics:
   - Level of trust and openness
   - Shared interests and experiences
   - Evolution of rapport over time
   - Communication preferences

Base your insights ONLY on the provided memories. DO NOT make assumptions or invent details.

Memories to analyze: {memories}

Respond with a JSON object using this exact structure (no other text):
{{{{
    "interests": [],        
    "preferences": [],      
    "behavioral_traits": [], 
    "emotional_patterns": [], 
    "recurring_themes": [], 
    "key_insights": [],
    "conversation_style": {{{{
        "formality_level": "",
        "humor_presence": "",
        "discussion_depth": "",
        "preferred_topics": []
    }}}},
    "emotional_context": {{{{
        "overall_tone": "",
        "significant_emotions": [],
        "emotional_triggers": [],
        "comfort_topics": []
    }}}}     
}}}}""")
        ])
        
        print(f"Memory template input variables: {self.memory_synthesis_prompt.input_variables}")
        print(f"Memory template message types: {[type(m) for m in self.memory_synthesis_prompt.messages]}")
        
        # Create response generation prompt that properly uses memories
        self.response_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a thoughtful AI assistant having a natural conversation. Maintain this tone:
- Conversation Style: {style}
- Emotional Context: {emotion}
- Response Approach: {approach}
- Formality Level: {formality}

Guidelines:
1. Keep responses natural and engaging
2. Be honest about what you remember
3. Share memories only when they add value to the conversation
4. NEVER make up memories
5. If memories aren't relevant, focus on the current conversation
6. When sharing memories, include specific details to maintain authenticity
7. Let the conversation flow naturally - don't force memory references

Remember: You have access to past conversations through the provided memories. Use them thoughtfully when they enrich the discussion."""),
            ("human", """Query: {input}
Analysis: {analysis}

Relevant Memories:
{memories}

Memory Insights: {synthesis}

Important: Focus on having a natural conversation. Use memories when they genuinely add value and help the discussion progress.""")
        ])
        
        # Add message-specific prompt storage
        self.message_prompts = {}  # Store prompts by message ID

    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine search strategy and context requirements."""
        try:
            # Streamlined analysis prompt
            analysis = ChatPromptTemplate.from_messages([
                ("system", """Analyze the query to extract key information for memory search.
Output a JSON object with:
{{
    "search_strategy": {{
        "primary_aspects": [],      # Main concepts to search for
        "semantic_variations": [],  # Related terms/phrases
        "temporal_scope": "",      # Time period if mentioned
        "context_type": ""         # Type of information needed
    }}  ,
    "required_context": {{
        "interests": false,         # Whether interests are relevant
        "preferences": false,       # Whether preferences matter
        "knowledge": false,         # Whether knowledge/facts needed
        "emotions": false          # Whether emotional context matters
    }},
    "tone_analysis": {{
        "conversation_style": "",    # e.g., "casual_chat", "formal_inquiry" 
        "emotional_context": "",     # e.g., "light_hearted", "reflective"
        "response_approach": "",     # e.g., "chat_naturally", "provide_details"
        "formality_level": ""       # e.g., "casual", "semi_formal"
    }}
}}"""),
                ("human", "{query}")
            ])
            
            chain = analysis | self.analysis_llm
            result = await chain.ainvoke({"query": query})
            
            # Extract and validate analysis
            if hasattr(result, 'content'):
                analysis_text = result.content
            elif isinstance(result, dict):
                analysis_text = result.get('content', str(result))
            else:
                analysis_text = str(result)
                
            try:
                # Parse JSON response
                analysis_dict = json.loads(analysis_text)
                
                # Validate required fields
                required_fields = ['search_strategy', 'required_context', 'tone_analysis']
                if not all(field in analysis_dict for field in required_fields):
                    print("Warning: Analysis missing required fields")
                    # Add defaults for missing fields
                    if 'search_strategy' not in analysis_dict:
                        analysis_dict['search_strategy'] = {"primary_aspects": [query]}
                    if 'required_context' not in analysis_dict:
                        analysis_dict['required_context'] = {"interests": False, "preferences": False}
                    if 'tone_analysis' not in analysis_dict:
                        analysis_dict['tone_analysis'] = {
                            "conversation_style": "casual_chat",
                            "emotional_context": "neutral",
                            "response_approach": "chat_naturally",
                            "formality_level": "casual"
                        }
                    elif isinstance(analysis_dict['tone_analysis'], str):
                        # Convert string tone_analysis to proper structure
                        tone_text = analysis_dict['tone_analysis']
                        analysis_dict['tone_analysis'] = {
                            "conversation_style": "casual_chat",
                            "emotional_context": "neutral",
                            "response_approach": "chat_naturally",
                            "formality_level": "casual"
                        }
                        print(f"Converted string tone_analysis: {tone_text}")
                        
                return analysis_dict
                
            except json.JSONDecodeError as e:
                print(f"Error parsing analysis JSON: {str(e)}")
                # Return basic analysis if parsing fails
                return {
                    "search_strategy": {
                        "primary_aspects": [query],
                        "semantic_variations": [],
                        "temporal_scope": "",
                        "context_type": "general"
                    },
                    "required_context": {
                        "interests": False,
                        "preferences": False,
                        "knowledge": False,
                        "emotions": False
                    },
                    "tone_analysis": {
                        "conversation_style": "casual_chat",
                        "emotional_context": "neutral",
                        "response_approach": "chat_naturally",
                        "formality_level": "casual"
                    }
                }
            
        except Exception as e:
            print(f"Error in query analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Return basic analysis on error
            return {
                "search_strategy": {"primary_aspects": [query]},
                "required_context": {"interests": False, "preferences": False},
                "tone_analysis": {
                    "conversation_style": "casual_chat",
                    "emotional_context": "neutral",
                    "response_approach": "chat_naturally",
                    "formality_level": "casual"
                }
            }

    async def _synthesize_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize insights from retrieved memories."""
        try:
            # Only synthesize if we have enough memories
            if not memories or len(memories) < 2:
                return {}
                
            # Streamlined synthesis prompt
            synthesis = ChatPromptTemplate.from_messages([
                ("system", """Analyze the provided memories to extract key insights.
Output a JSON object with:
- key_themes: list of main themes
- emotional_context: overall emotional tone
- notable_patterns: any recurring patterns"""),
                ("human", "Analyze these memories: {memories}")
            ])
            
            chain = synthesis | self.synthesis_llm
            result = await chain.ainvoke({
                "memories": json.dumps([{
                    "text": m.get("text", ""),
                    "timestamp": m.get("timestamp", "")
                } for m in memories], ensure_ascii=False)
            })
            
            # Extract and validate synthesis
            if hasattr(result, 'content'):
                synthesis_text = result.content
            elif isinstance(result, dict):
                synthesis_text = result.get('content', str(result))
            else:
                synthesis_text = str(result)
            
            try:
                # Parse JSON response
                synthesis_dict = json.loads(synthesis_text)
                
                # Validate required fields
                required_fields = ['key_themes', 'emotional_context', 'notable_patterns']
                if not all(field in synthesis_dict for field in required_fields):
                    print("Warning: Synthesis missing required fields")
                    # Add defaults for missing fields
                    if 'key_themes' not in synthesis_dict:
                        synthesis_dict['key_themes'] = []
                    if 'emotional_context' not in synthesis_dict:
                        synthesis_dict['emotional_context'] = "neutral"
                    if 'notable_patterns' not in synthesis_dict:
                        synthesis_dict['notable_patterns'] = []
                    
                return synthesis_dict
                
            except json.JSONDecodeError as e:
                print(f"Error parsing synthesis JSON: {str(e)}")
                # Return basic synthesis if parsing fails
                return {
                    "key_themes": [],
                    "emotional_context": "neutral",
                    "notable_patterns": []
                }
            
        except Exception as e:
            print(f"Error in memory synthesis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            # Return empty synthesis on error
            return {}

    def _semantic_retriever(self, query: str, analysis: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve results based on semantic similarity."""
        results = []
        try:
            query_embedding = self.embeddings_manager.get_embedding(query)
            if query_embedding:
                base_results = self.vector_store.search_similar(
                    query_vector=query_embedding,
                    top_k=top_k,
                    min_similarity=0.5
                )
                if base_results:
                    results.extend(self._verify_and_format_results(base_results))
        except Exception as e:
            print(f"Error in semantic retriever: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        return results

    def _temporal_retriever(self, query: str, analysis: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve results based on temporal relevance."""
        results = []
        temporal_scope = analysis.get('search_strategy', {}).get('temporal_scope', '')
        if not temporal_scope:
            return results

        try:
            # Create a temporal query embedding
            temporal_query = f"Conversation from {temporal_scope}"
            temporal_embedding = self.embeddings_manager.get_embedding(temporal_query)
            
            if temporal_embedding:
                # Get potential matches using vector similarity
                temporal_results = self.vector_store.search_similar(
                    query_vector=temporal_embedding,
                    top_k=100,  # Get more results to filter
                    min_similarity=0.3  # Lower threshold for temporal matches
                )
                
                if temporal_results:
                    # Post-process results to score temporal relevance
                    for result in temporal_results:
                        timestamp = result.get('timestamp', '')
                        if not timestamp:
                            continue
                        
                        try:
                            conv_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            temporal_score = self._calculate_temporal_score(conv_date, temporal_scope)
                            if temporal_score > 0:
                                result['temporal_score'] = temporal_score
                                results.append(result)
                        except ValueError:
                            continue

                    # Sort by temporal score and take top_k
                    results.sort(key=lambda x: x.get('temporal_score', 0), reverse=True)
                    results = results[:top_k]

            print(f"Temporal retriever found {len(results)} results for scope: {temporal_scope}")
            return self._verify_and_format_results(results)
            
        except Exception as e:
            print(f"Error in temporal retriever: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

    def _tonal_retriever(self, query: str, analysis: Dict[str, Any], top_k: int = 20) -> List[Dict[str, Any]]:
        """Retrieve results based on conversational tone and style."""
        results = []
        tone_analysis = analysis.get('tone_analysis', {})
        style = tone_analysis.get('conversation_style', '')
        emotion = tone_analysis.get('emotional_context', '')
        
        if not (style or emotion):
            return results

        # Create tone-focused query
        tone_query = f"Conversation showing {style} style and {emotion} emotional context"
        tone_embedding = self.embeddings_manager.get_embedding(tone_query)
        if tone_embedding:
            tone_results = self.vector_store.search_similar(
                query_vector=tone_embedding,
                top_k=top_k,
                min_similarity=0.6
            )
            if tone_results:
                results.extend(self._verify_and_format_results(tone_results))
        return results

    def _calculate_temporal_score(self, conv_date: datetime, temporal_scope: str) -> float:
        """Calculate temporal relevance score based on conversation date and query scope."""
        now = datetime.now()
        
        # Extract time indicators from temporal scope
        is_around = "around" in temporal_scope.lower()
        window_size = timedelta(weeks=4 if is_around else 2)
        
        # Calculate base score based on temporal distance
        if "recent" in temporal_scope.lower():
            target_date = now
        else:
            # For specific dates/holidays, you would parse them here
            # For now, using a simple approach
            target_date = conv_date
        
        time_diff = abs((conv_date - target_date).total_seconds())
        max_diff = window_size.total_seconds()
        
        if time_diff > max_diff:
            return 0.0
        
        # Score decreases linearly with distance from target date
        return 1.0 - (time_diff / max_diff)

    def _search_conversations(self, query: str, analysis: Dict[str, Any], max_results: int = 20) -> List[Dict[str, Any]]:
        """Execute multi-retriever search and evaluate results holistically."""
        try:
            print(f"Executing multi-retriever search for: {query}")
            
            # Get candidate pool from all retrievers
            semantic_results = self._semantic_retriever(query, analysis)
            temporal_results = self._temporal_retriever(query, analysis)
            tonal_results = self._tonal_retriever(query, analysis)
            
            # Combine all results, removing duplicates
            all_results = []
            seen_ids = set()
            
            for result in semantic_results + temporal_results + tonal_results:
                if result['conversation_id'] not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result['conversation_id'])
            
            print(f"Found {len(all_results)} total candidate results")
            
            # Create evaluation chain
            evaluation_prompt = ChatPromptTemplate.from_messages([
                ("system", '''You are having a natural, friendly conversation with someone. When they ask about specific conversations or memories, keep the same warm, casual tone you've been using.

Your task is to evaluate each conversation for:

1. How well it matches what we're chatting about:
   - Is this the specific memory they're thinking of?
   - Does it add something interesting to our chat?
   - Would it help continue our conversation naturally?

2. When it happened:
   - If they mention a specific time, is this from then?
   - For "around" a certain time, is it close enough?
   - Would mentioning this timing feel natural in our chat?

3. The feeling and style:
   - Does this match our current conversation's vibe?
   - Would sharing this maintain our chat's flow?
   - Would this feel like a natural thing to bring up?

4. Overall fit:
   - Would mentioning this enrich our conversation?
   - Would it help us connect over shared memories?
   - Would it feel natural to bring up?

Think about this conversation:
Query: {query}
Analysis: {analysis}
Timestamp: {timestamp}
Content: {content}

Respond with a JSON object using this exact structure (no other text):
{{"relevance": {{"semantic_match": 0.95, "temporal_match": 0.85, "tonal_match": 0.75, "overall_score": 0.85, "reasoning": "This is exactly what we were talking about, and it happened just when they mentioned"}}}}''')
            ])
            
            evaluation_chain = evaluation_prompt | self.analysis_llm | JsonOutputParser()
            
            # Evaluate each result
            evaluated_results = []
            for result in all_results:
                try:
                    eval_input = {
                        "query": query,
                        "analysis": json.dumps(analysis, indent=2),
                        "timestamp": result.get('timestamp', ''),
                        "content": result.get('text', '')
                    }
                    
                    print(f"\nEvaluating result {result['conversation_id']}")
                    evaluation = evaluation_chain.invoke(eval_input)
                    print(f"Evaluation successful: {evaluation}")
                    
                    # Add evaluation scores to result
                    result.update(evaluation['relevance'])
                    evaluated_results.append(result)
                    
                except Exception as chain_error:
                    print("\nError in evaluation chain:")
                    print(f"Error type: {type(chain_error).__name__}")
                    print(f"Error message: {str(chain_error)}")
                    import traceback
                    print("\nFull stack trace:")
                    print(traceback.format_exc())
                    print("\nEvaluation input that caused error:")
                    for key, value in eval_input.items():
                        print(f"\n{key}:")
                        print(value)
            
            # Sort by overall score
            evaluated_results.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
            
            print(f"Evaluation complete. Returning top {min(max_results, len(evaluated_results))} results")
            return evaluated_results[:max_results]
            
        except Exception as e:
            print(f"Error in conversation search: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return []

    def _filter_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Filter results using AI evaluation."""
        try:
            print("\n=== Starting AI Filtering ===")
            print(f"Number of results to evaluate: {len(results)}")
            print(f"Original query: {query}")
            
            filtered_results = []
            
            print("Creating evaluation chain...")
            # Create a simpler template for testing
            evaluation_prompt = ChatPromptTemplate.from_messages([
                ("system", '''You are an expert at evaluating the relevance of conversation snippets to a user's query, with a strong focus on temporal relevance. Your task is to carefully analyze each conversation and determine if it's truly relevant to what the user is asking about.

Consider these key aspects in order of importance:

1. Temporal Relevance (CRITICAL):
   - Analyze the query's temporal scope:
     * Is there a specific time period mentioned?
     * What does "around" mean in this context?
     * How precise does the time match need to be?
   
   - Consider relative time periods:
     * When someone says "around" a date, it usually implies ±2-4 weeks
     * The more significant the event/date, the wider the relevant time window
     * Recent dates may need more precision than older ones
   
   - Evaluate temporal proximity:
     * "exact" = The specific time period mentioned
     * "close" = Within the broader timeframe implied by "around"
     * "related" = Within the general season or period
     * "unrelated" = Clearly outside the relevant timeframe

2. Content Relevance:
   - Does this conversation actually answer what the user is asking?
   - Is it directly related or only tangentially related?
   - Does it provide valuable context or insight?
   - For personal queries about "what was on my mind":
     * Does it reveal thoughts, concerns, or interests?
     * Does it show emotional state or mindset?
     * Does it capture activities or plans from that time?

3. Query Intent Match:
   - Does this match what the user is really trying to learn?
   - Is it aligned with the style of information they're seeking?
   - Would this help create the kind of response they're looking for?

IMPORTANT: The meaning of "around" depends on context - consider the significance of the event/date and how precise the user seems to want the time match to be.

Respond with a JSON object using this exact structure (no other text):
{{
    "is_relevant": true,
    "relevance_score": 0.85,
    "temporal_match": "close",
    "content_match": "direct",
    "reasoning": "Conversation is within the broader timeframe implied by 'around' in the query"
}}'''),
                ("human", "Query: {query}\nAnalysis: {analysis}\nTimestamp: {timestamp}\nContent: {content}")
            ])
            
            print("\nDebug: Template Analysis")
            print(f"Template type: {type(evaluation_prompt)}")
            print(f"Template messages: {evaluation_prompt.messages}")
            print(f"Raw input variables: {evaluation_prompt.input_variables}")
            
            evaluation_chain = evaluation_prompt | self.analysis_llm | JsonOutputParser()
            
            print("\nDebug: Evaluation Chain Structure")
            print(f"Chain components: {[type(component).__name__ for component in evaluation_chain.steps]}")
            
            for i, result in enumerate(results, 1):
                print(f"\nEvaluating result {i}/{len(results)}")
                
                eval_input = {
                    "query": query,  # Use the original query instead of the result text
                    "analysis": json.dumps(analysis, indent=2),
                    "timestamp": result.get('timestamp', ''),
                    "content": result.get('text', '')
                }
                
                print("\nDebug: Evaluation Input")
                print(f"Original query: {query}")
                print(f"Keys provided: {list(eval_input.keys())}")
                print(f"Expected keys: {evaluation_prompt.input_variables}")
                print(f"Query length: {len(eval_input['query'])}")
                print(f"Analysis length: {len(eval_input['analysis'])}")
                print(f"Content sample: {eval_input['content'][:100]}...")
                
                try:
                    print("\nAttempting evaluation...")
                    eval_result = evaluation_chain.invoke(eval_input)
                    print(f"Evaluation successful: {eval_result}")
                    
                    if eval_result.get('is_relevant', False):
                        result.update(eval_result)
                        filtered_results.append(result)
                        
                except Exception as chain_error:
                    print("\nError in evaluation chain:")
                    print(f"Error type: {type(chain_error).__name__}")
                    print(f"Error message: {str(chain_error)}")
                    import traceback
                    print("\nFull stack trace:")
                    print(traceback.format_exc())
                    print("\nEvaluation chain components:")
                    print(f"Chain steps: {[type(component).__name__ for component in evaluation_chain.steps]}")
                    print("\nEvaluation input that caused error:")
                    for key, value in eval_input.items():
                        print(f"\n{key}:")
                        print(value)
            
            filtered_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            print(f"\nFiltering complete. Found {len(filtered_results)} relevant results")
            return filtered_results
            
        except Exception as e:
            print("\nError in AI filtering:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
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

    def _should_update_tone(self, new_tone: Dict[str, str], message_type: str = "normal") -> Tuple[bool, float]:
        """Determine if tone should be updated based on current context and history."""
        if not self.tone_history:
            return True, 1.0
            
        # Calculate tone difference score
        difference_score = 0
        total_weight = 0
        
        for key, weight in self.tone_weights.items():
            if key in new_tone and key in self.conversation_tone:
                current = self.conversation_tone[key]
                proposed = new_tone[key]
                
                # Higher weight for drastic changes
                if any(signal in proposed for signal in ['formal', 'serious', 'emotional', 'technical']):
                    weight *= 1.5
                
                # Check for tone continuity
                if current != proposed:
                    difference_score += weight
                total_weight += weight
        
        normalized_difference = difference_score / total_weight if total_weight > 0 else 0
        
        # Adjust threshold based on message type
        adjusted_threshold = self.tone_transition_threshold
        if message_type == "question":
            adjusted_threshold *= 0.8  # More willing to change for questions
        elif message_type == "emotional":
            adjusted_threshold *= 0.6  # Even more willing to change for emotional content
        
        # Consider recent tone history
        if len(self.tone_history) >= 2:
            recent_changes = sum(1 for i in range(len(self.tone_history)-1) 
                               if self.tone_history[i] != self.tone_history[i+1])
            if recent_changes >= 2:
                adjusted_threshold *= 1.2  # More resistant to change if tone has been changing frequently
        
        return normalized_difference > adjusted_threshold, normalized_difference

    def _update_conversation_tone(self, new_tone: Dict[str, str], message_type: str = "normal"):
        """Update conversation tone with smoother transitions."""
        should_update, difference_score = self._should_update_tone(new_tone, message_type)
        
        if should_update:
            # Store current tone in history
            self.tone_history.append(self.conversation_tone.copy())
            if len(self.tone_history) > 5:  # Keep last 5 tones
                self.tone_history.pop(0)
            
            # Update tone
            for key, value in new_tone.items():
                if key in self.conversation_tone:
                    self.conversation_tone[key] = value
            
            print(f"Tone updated (difference: {difference_score:.2f})")
            print(f"New tone: {self.conversation_tone}")
        else:
            print(f"Tone maintained (difference: {difference_score:.2f} below threshold)")
            
        return should_update

    def _verify_conversation_participants(self, text: str) -> bool:
        """Verify that the conversation involves the user and assistant."""
        # Simple check for now - could be enhanced with more sophisticated parsing
        return True  # Accept all conversations for now

    def _format_conversation_snippet(self, text: str, timestamp: str = '') -> str:
        """Format a conversation snippet with timestamp."""
        formatted_text = text.strip()
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_timestamp = dt.strftime('%B %d, %Y')
                formatted_text = f"[{formatted_timestamp}]\n\n{formatted_text}"
            except ValueError:
                # If timestamp parsing fails, just use the raw timestamp
                formatted_text = f"[{timestamp}]\n\n{formatted_text}"
        return formatted_text

    async def process_message(self, message: str, conversation_id: Optional[str] = None) -> dict:
        """Process a message and return the response with context."""
        print(f"\n=== Processing Chat Request ===")
        print(f"Message: {message}")
        try:
            message_id = str(uuid.uuid4())
            
            # Stage 1: Query Analysis
            query_analysis = await self._analyze_query(message)
            print(f"Query analysis: {query_analysis}")
            
            # Update conversation tone if needed
            message_type = "normal"
            if "?" in message:
                message_type = "question"
            elif any(word in message.lower() for word in ['feel', 'sad', 'happy', 'angry', 'love', 'hate']):
                message_type = "emotional"
            
            if 'tone_analysis' in query_analysis:
                print(f"Updating tone with: {query_analysis['tone_analysis']}")
                self._update_conversation_tone(query_analysis['tone_analysis'], message_type)
            
            # Stage 2: Memory Retrieval
            search_results = self._search_conversations(message, query_analysis, max_results=10)
            unique_results = []
            if search_results:
                # Only keep highly relevant results
                unique_results = [r for r in search_results if r.get('overall_score', 0) > 0.8]
                unique_results = unique_results[:5]  # Keep only top 5 most relevant
                print(f"Filtered to {len(unique_results)} highly relevant results")
            
            # Stage 3: Memory Synthesis
            synthesis = {}
            if len(unique_results) >= 2:
                synthesis = await self._synthesize_memories(unique_results)
            
            # Format memories into a clear, readable format
            formatted_memories = []
            for result in unique_results:
                formatted_memory = f"""
Timestamp: {result.get('timestamp', '')}
Conversation:
{result.get('text', '')}
Relevance: {result.get('overall_score', 0)} - {result.get('reasoning', '')}

"""
                formatted_memories.append(formatted_memory)
            
            memories_text = "No relevant memories found" if not formatted_memories else "\n".join(formatted_memories)
            
            # Stage 4: Response Generation
            response = ChatPromptTemplate.from_messages([
                ("system", """You are a thoughtful AI assistant having a natural conversation. Maintain this tone:
- Conversation Style: {style}
- Emotional Context: {emotion}
- Response Approach: {approach}
- Formality Level: {formality}

Guidelines:
1. Keep responses natural and engaging
2. Be honest about what you remember
3. Share memories only when they add value to the conversation
4. NEVER make up memories
5. If memories aren't relevant, focus on the current conversation
6. When sharing memories, include specific details to maintain authenticity
7. Let the conversation flow naturally - don't force memory references

Remember: You have access to past conversations through the provided memories. Use them thoughtfully when they enrich the discussion."""),
                ("human", """Query: {input}
Analysis: {analysis}

Relevant Memories:
{memories}

Memory Insights: {synthesis}

Important: Focus on having a natural conversation. Use memories when they genuinely add value and help the discussion progress.""")
            ])
            
            # Debug: Print the full formatted prompt
            print("\n=== Response Generation Prompt ===")
            debug_input = {
                "input": message,
                "analysis": json.dumps(query_analysis, ensure_ascii=False),
                "memories": memories_text,
                "synthesis": json.dumps(synthesis, ensure_ascii=False) if synthesis else "No synthesis performed",
                "style": self.conversation_tone['conversation_style'],
                "emotion": self.conversation_tone['emotional_context'],
                "approach": self.conversation_tone['response_approach'],
                "formality": self.conversation_tone['formality_level']
            }
            
            formatted_prompt = response.format_messages(**debug_input)
            print("\nSystem Message:")
            print(formatted_prompt[0].content)
            print("\nHuman Message:")
            print(formatted_prompt[1].content)
            print("\n=== End Response Generation Prompt ===\n")
            
            chain = response | self.response_llm
            result = await chain.ainvoke(debug_input)
            
            # Extract and format response
            if hasattr(result, 'content'):
                response_text = result.content
            elif isinstance(result, dict) and 'content' in result:
                response_text = result['content']
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)
                if response_text.startswith('messages=[') and ']' in response_text:
                    try:
                        ai_msg_start = response_text.rfind("AIMessage(content='") + 19
                        ai_msg_end = response_text.rfind("')")
                        if ai_msg_start > 18 and ai_msg_end > ai_msg_start:
                            response_text = response_text[ai_msg_start:ai_msg_end]
                    except Exception as e:
                        print(f"Error extracting AI message: {str(e)}")
                        response_text = "I apologize, but I encountered an error processing your request."
            
            # Debug info
            debug_info = {
                "stages": {
                    "query_analysis": query_analysis,
                    "memory_retrieval": {
                        "results_count": len(unique_results),
                        "top_result_score": unique_results[0].get("overall_score", 0) if unique_results else 0
                    },
                    "synthesis_performed": bool(synthesis),
                    "current_tone": self.conversation_tone
                },
                "performance": {
                    "timestamp": datetime.now().isoformat(),
                    "conversation_id": conversation_id,
                    "message_id": message_id
                }
            }
            
            return {
                "response": response_text,
                "context_used": bool(unique_results),
                "synthesis_performed": bool(synthesis),
                "conversation_id": conversation_id,
                "message_id": message_id,
                "debug_info": debug_info
            }
            
        except Exception as e:
            print("\nError in process_message:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            print("\nFull traceback:")
            print(traceback.format_exc())
            return {
                "response": "I encountered an error while processing your message.",
                "context_used": False,
                "error": str(e),
                "conversation_id": conversation_id,
                "message_id": str(uuid.uuid4()),
                "debug_info": {
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    }
                }
            }

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
        print(f"Processing chat request: {message}")
        try:
            # Delegate to memory agent
            return await self.memory_agent.process_message(message, conversation_id)
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "response": "I encountered an error while processing your message.",
                "context_used": False,
                "error": str(e),
                "conversation_id": conversation_id,
                "message_id": str(uuid.uuid4()),
                "debug_info": {
                    "error": {
                        "type": type(e).__name__,
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    }
                }
            }
