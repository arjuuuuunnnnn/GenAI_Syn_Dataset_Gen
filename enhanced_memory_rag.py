from typing import Dict, List, Any, Optional
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from data_memory import GeneratedDataMemory
from conversation_memory import ConversationMemory

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)

class EnhancedMemoryRAG:
    """Enhanced RAG system that incorporates knowledge, data memory, and conversation history"""
    
    def __init__(self):
        os.makedirs("./knowledge_docs", exist_ok=True)
        os.makedirs("./chroma_db", exist_ok=True)
        
        try:
            print("Loading embedding model...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Static knowledge base
            self.db = chromadb.PersistentClient(path="./chroma_db")
            self.knowledge_collection = self.db.get_or_create_collection("knowledge_base")
            
            # Memory systems
            self.data_memory = GeneratedDataMemory()
            self.conversation_memory = ConversationMemory()
            
            # Initialize LLM
            self.llm = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize knowledge base if empty
            if len(self.knowledge_collection.get()['ids']) == 0:
                self._initialize_knowledge_base()
            
            print("Enhanced memory RAG system initialized successfully.")
        except Exception as e:
            print(f"Warning: Enhanced RAG system initialization failed: {str(e)}")
            print("The system will function with limited capabilities.")
            self.embed_model = None
            self.db = None
            self.data_memory = None
            self.conversation_memory = None
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with system documentation"""
        # Same as before, unchanged
        print("Initializing knowledge base...")
        # [Knowledge base initialization code]
    
    def identify_query_type(self, query: str) -> Dict:
        """Identify if a query is about generation, memory retrieval, or referencing past conversations"""
        try:
            # First try to retrieve any relevant conversation history
            conv_results = self.conversation_memory.search_conversation_history(query, limit=1)
            has_relevant_conversation = bool(conv_results)
            
            # Check if this is likely a follow-up question
            follow_up_indicators = [
                "previous", "before", "earlier", "you said", "you mentioned",
                "you told", "last time", "last response", "that", "those", "it", "them"
            ]
            
            is_likely_followup = any(indicator in query.lower() for indicator in follow_up_indicators)
            
            # Use LLM to analyze query type
            prompt = f"""
            Analyze this user query and determine if it's:
            1. Requesting new text/data generation
            2. Requesting new image generation
            3. Referring to previous conversations (follow-up question)
            4. Asking about previously generated content (memory)
            
            Additional context:
            {"- The query appears to relate to a previous conversation." if has_relevant_conversation else ""}
            {"- The query contains follow-up indicators like 'previous', 'last time', etc." if is_likely_followup else ""}
            
            Return JSON with this format:
            {{
                "query_type": "generation" or "memory" or "conversation_reference",
                "generation_type": "text" or "image" or null,
                "memory_focus": "text" or "image" or "general" or null,
                "is_followup": true or false,
                "confidence": 0-1 value,
                "explanation": "brief explanation"
            }}
            
            User Query: {query}
            """
            
            response = self.llm.generate_content(prompt)
            try:
                analysis = json.loads(response.text)
                
                # Override with conversation reference if we have a relevant conversation and it's likely a follow-up
                if has_relevant_conversation and (is_likely_followup or analysis.get("is_followup", False)):
                    analysis["query_type"] = "conversation_reference"
                    analysis["confidence"] = max(analysis.get("confidence", 0.5), 0.8)
                    analysis["explanation"] = "Query appears to reference previous conversation."
                
                return analysis
            except:
                # If parsing fails, use keyword matching as fallback
                pass
        except Exception as e:
            print(f"Error in query type identification: {str(e)}")
        
        # Fallback logic based on keywords
        # ...similar to before but with additional conversation reference detection
        
        memory_keywords = ["previous", "generated", "created", "last time", "remember", 
                          "recall", "show me", "tell me about", "what was", "memory"]
        conversation_keywords = ["you said", "you mentioned", "you told", "earlier",
                               "last response", "before", "that", "those", "it", "them"]
        text_keywords = ["data", "list", "table", "dataset", "text", "write", "create content"]
        image_keywords = ["image", "picture", "photo", "draw", "visual", "art", "illustration"]

        content_question_patterns = ["what is", "how many", "tell me about", "analyze", "calculate", "summarize", "find", "list", "show me", "what are", "which"]
        
        query_lower = query.lower()
        
        # Count keywords in each category
        memory_count = sum(1 for kw in memory_keywords if kw in query_lower)
        conversation_count = sum(1 for kw in conversation_keywords if kw in query_lower)
        text_count = sum(1 for kw in text_keywords if kw in query_lower)
        image_count = sum(1 for kw in image_keywords if kw in query_lower)

        has_content_question = any(pattern in query.lower() for pattern in content_question_patterns)

        if has_content_question:
            memory_result = self.search_data_memory(query)
            if memory_result["success"]:
                return {
                    "query_type": "content_question",
                    "memory_focus": memory_result["top_result"]["type"],
                    "is_followup": True,
                    "confidence": 0.85,
                    "explanation": "Query appears to be asking about the content of previously generated data."
                }
        
        # Check for conversation reference first
        if conversation_count > 0 or has_relevant_conversation:
            return {
                "query_type": "conversation_reference",
                "generation_type": None,
                "memory_focus": None,
                "is_followup": True,
                "confidence": 0.75,
                "explanation": "Query appears to reference previous conversation."
            }
        # Then check memory
        elif memory_count > max(text_count, image_count):
            memory_focus = "general"
            if text_count > image_count:
                memory_focus = "text"
            elif image_count > text_count:
                memory_focus = "image"
            
            return {
                "query_type": "memory",
                "generation_type": None,
                "memory_focus": memory_focus,
                "is_followup": False,
                "confidence": 0.7,
                "explanation": "Query contains memory-related keywords."
            }
        # Then check generation types
        elif text_count > image_count:
            return {
                "query_type": "generation",
                "generation_type": "text",
                "memory_focus": None,
                "is_followup": False,
                "confidence": 0.7,
                "explanation": "Query contains text generation keywords."
            }
        elif image_count > text_count:
            return {
                "query_type": "generation",
                "generation_type": "image",
                "memory_focus": None,
                "is_followup": False,
                "confidence": 0.7,
                "explanation": "Query contains image generation keywords."
            }
        else:
            return {
                "query_type": "unknown",
                "generation_type": None,
                "memory_focus": None,
                "is_followup": False,
                "confidence": 0.5,
                "explanation": "Unable to determine query type with confidence."
            }
    
    def store_interaction(self, user_query: str, system_response: str, agent_type: str) -> str:
        """Store a conversation interaction for future reference"""
        if not self.conversation_memory:
            return None
        
        try:
            return self.conversation_memory.store_interaction(
                user_query=user_query,
                system_response=system_response,
                agent_type=agent_type
            )
        except Exception as e:
            print(f"Error storing interaction: {str(e)}")
            return None
    
    def retrieve_conversation_context(self, query: str) -> Dict:
        """Retrieve relevant previous conversation context"""
        if not self.conversation_memory:
            return {
                "success": False,
                "reason": "Conversation memory system unavailable"
            }
    
        try:
            # Check if this is the first interaction
            try:
                conversations = self.conversation_memory.search_conversation_history(query, limit=2)
            except Exception as e:
                print(f"Error searching conversation history: {str(e)}")
                # If there's an error with searching (possibly no conversations yet)
                return {
                    "success": False,
                    "reason": "No conversation history available yet",
                    "error": str(e)
                }
        
            if conversations and len(conversations) > 0:
                return {
                    "success": True,
                    "conversations": conversations,
                    "summary": f"Found {len(conversations)} relevant conversation(s)"
                }
            else:
                return {
                    "success": False,
                    "reason": "No relevant conversation history found"
                }
        except Exception as e:
            print(f"Error retrieving conversation context: {str(e)}")
            return {
                "success": False,
                "reason": f"Error: {str(e)}"
            }
    
    def answer_with_conversation_context(self, query: str) -> str:
        """Generate an answer that incorporates conversation history"""
        context_result = self.retrieve_conversation_context(query)
    
        if not context_result["success"]:
            # No relevant conversation, try data memory
            memory_result = self.search_data_memory(query)
            if memory_result["success"]:
                return self.answer_memory_query(query)
            else:
                # Handle the case where this is the first interaction
                if "No relevant conversation history found" in context_result.get("reason", ""):
                    return "I don't have any previous conversations stored yet. This appears to be our first interaction. Feel free to ask me something else, and I'll remember our conversation for future reference."
                return f"I don't have any relevant conversation history or memory to answer your query about previous interactions."
    
        # We have relevant conversation history
        conversations = context_result["conversations"]
    
        # Make sure we have at least one conversation before trying to access it
        if not conversations or len(conversations) == 0:
            return "I found some conversation records, but couldn't retrieve their content. Let's start fresh with a new topic."
    
        # Extract the most relevant conversation
        conv = conversations[0]
        prev_query = conv["query"]
        prev_response = conv["response"]
    
        # Generate a response that incorporates the previous conversation
        prompt = f"""
        User previously asked: "{prev_query}"
    
        My previous response was: "{prev_response}"
    
        Now the user is asking a follow-up question: "{query}"
    
        Provide a helpful response that maintains continuity with the previous conversation.
        If this isn't actually a follow-up question, just answer it directly.
        """
    
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response with conversation context: {str(e)}")
            # Fall back to the previous response as context
            return f"Based on our previous conversation about '{prev_query}', I can tell you that {prev_response[:200]}... Does that help answer your current question?"

    def query_knowledge_base(self, query: str) -> str:
        """Query the static knowledge base"""
        # Same as before, unchanged
        
    def search_data_memory(self, query: str) -> Dict:
        """Search the data memory system for relevant content"""
        # This was previously called search_memory in memory_rag.py
        if not self.data_memory:
            return {
                "success": False,
                "reason": "Data memory system unavailable",
                "results": []
            }
        
        try:
            # Check if query contains a memory ID
            memory_id = None
            query_lower = query.lower()
            
            # Look for memory ID mentions
            if "id" in query_lower and any(prefix in query_lower for prefix in ["text_", "img_"]):
                words = query_lower.split()
                for word in words:
                    if word.startswith("text_") or word.startswith("img_"):
                        memory_id = word
                        break
            
            # If memory ID is found, retrieve that specific content
            if memory_id:
                memory_data = self.data_memory.get_data_by_id(memory_id)
                if memory_data:
                    return {
                        "success": True,
                        "query_type": "specific_id",
                        "memory_id": memory_id,
                        "results": [memory_data],
                        "summary": self.data_memory.get_human_readable_memory_summary(memory_data)
                    }
                else:
                    return {
                        "success": False,
                        "query_type": "specific_id",
                        "reason": f"Memory with ID {memory_id} not found",
                        "results": []
                    }
            
            # Otherwise, search semantically
            results = self.data_memory.search_memory(query)
            
            if results and len(results) > 0:
                # Get full data for top result
                top_result_id = results[0]["id"]
                top_result_data = self.data_memory.get_data_by_id(top_result_id)
                
                if top_result_data:
                    memory_summary = self.data_memory.get_human_readable_memory_summary(top_result_data)
                    
                    return {
                        "success": True,
                        "query_type": "semantic_search",
                        "results": results,
                        "top_result": top_result_data,
                        "summary": memory_summary
                    }
            
            # If we get here, no relevant results
            recent_data = self.data_memory.get_recent_data(limit=3)
            
            if recent_data and len(recent_data) > 0:
                return {
                    "success": False,
                    "query_type": "semantic_search",
                    "reason": "No relevant memory matches found",
                    "recent_items": recent_data,
                    "results": []
                }
            else:
                return {
                    "success": False,
                    "query_type": "semantic_search",
                    "reason": "No memory data available",
                    "results": []
                }
                
        except Exception as e:
            print(f"Error searching memory: {str(e)}")
            return {
                "success": False,
                "reason": f"Error searching memory: {str(e)}",
                "results": []
            }
    
        def answer_memory_query(self, query: str) -> str:
            """Generate a comprehensive answer to a memory-related query"""
            memory_result = self.search_data_memory(query)
    
            if not memory_result["success"]:
                # No memory matches
                # Existing code for handling no matches...
    
            # We found memory matches
            if memory_result["query_type"] == "specific_id":
                # User asked about specific memory ID
                memory_data = memory_result["results"][0]
        
                # Detect if this is a general metadata question or a content-specific question
                content_question_patterns = ["what is", "how many", "analyze", "calculate", "find"]
                is_content_question = any(pattern in query.lower() for pattern in content_question_patterns)
        
                if is_content_question and memory_data.get("type") == "text" and "data" in memory_data:
                    # This is a question about the content of the data
                    actual_data = memory_data["data"]
                    prompt = f"""
                    Based on this data: {json.dumps(actual_data)}
            
                    Answer this question about the data: {query}
            
                    Provide a detailed analysis based only on the given data.
                    """
            
                    try:
                        response = self.llm.generate_content(prompt)
                        return f"Based on the data '{memory_data.get('description')}', {response.text}"
                    except Exception as e:
                        print(f"Error generating content-based response: {str(e)}")
                        # Fall back to metadata summary
                        return f"Here's the information about memory {memory_data.get('id', 'unknown')}:\n\n{memory_result['summary']}"
                else:
                    # Metadata question - use existing code
                    summary = memory_result["summary"]
                    return f"Here's the information about memory {memory_data.get('id', 'unknown')}:\n\n{summary}"


    def enhance_generation_query(self, query: str, query_type: str) -> Dict:
        """Enhance a generation query with relevant knowledge, memory context, and conversation history"""
        knowledge_context = self.query_knowledge_base(query)
        
        # For generation queries, add relevant memory content if helpful
        memory_search = self.search_data_memory(query)
        memory_context = ""
        
        if memory_search["success"]:
            # Found relevant memory that might be useful as context
            top_result = memory_search["top_result"]
            memory_type = top_result.get("type", "unknown")
            
            # Only add memory context if it's relevant to the current generation type
            if (query_type == "text" and memory_type == "text") or (query_type == "image" and memory_type == "image"):
                if "description" in top_result:
                    memory_context = f"Previous {memory_type} generation: {top_result['description']}"
        
        # Also check for relevant conversation history
        conversation_context = ""
        conversation_result = self.retrieve_conversation_context(query)
        if conversation_result["success"]:
            # Found relevant conversation history
            conversations = conversation_result["conversations"]
            if conversations and len(conversations) > 0:
                conv = conversations[0]
                conversation_context = f"Previous interaction - Query: {conv['query']} Response: {conv['response'][:200]}..."
        
        return {
            "enhanced_query": query,
            "knowledge_context": knowledge_context,
            "memory_context": memory_context,
            "conversation_context": conversation_context
        }
