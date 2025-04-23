from typing import TypedDict, Dict, List, Any, Optional
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from data_memory import GeneratedDataMemory

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)

class MemoryAwareRAG:
    """Enhanced RAG system that incorporates both static knowledge and dynamic memory"""
    
    def __init__(self):
        os.makedirs("./knowledge_docs", exist_ok=True)
        os.makedirs("./chroma_db", exist_ok=True)
        
        try:
            print("Loading embedding model...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Static knowledge base
            self.db = chromadb.PersistentClient(path="./chroma_db")
            self.knowledge_collection = self.db.get_or_create_collection("knowledge_base")
            
            # Memory system
            self.memory_system = GeneratedDataMemory()
            
            # Initialize LLM
            self.llm = genai.GenerativeModel('gemini-2.0-flash')
            
            # Initialize knowledge base if empty
            if len(self.knowledge_collection.get()['ids']) == 0:
                self._initialize_knowledge_base()
            
            print("Memory-aware RAG system initialized successfully.")
        except Exception as e:
            print(f"Warning: RAG system initialization failed: {str(e)}")
            print("The system will function with limited capabilities.")
            self.embed_model = None
            self.db = None
            self.memory_system = None
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with system documentation"""
        print("Initializing knowledge base...")
        
        # System documentation
        knowledge_content = {
            "generation_types.txt": """
            The system supports two types of generation:
            1. Text Generation: For creating datasets, lists, stories, and textual content
            2. Image Generation: For creating visual artwork, designs, and illustrations
            
            Text generation keywords: data, list, table, generate text, create content, dataset
            Image generation keywords: image, picture, photo, draw, visual, art, illustration
            
            The system can also answer questions about previously generated content by searching the memory system.
            Memory-related keywords: previous, generated, created, last time, remember, recall, show me
            """,
            
            "capabilities.txt": """
            System capabilities include:
            - Generating realistic dataset examples with various fields and constraints
            - Creating fictional character profiles and data collections
            - Producing landscape and abstract images
            - Designing visual concepts and illustrations
            - Remembering previously generated content for follow-up questions
            - Answering questions about previously generated data or images
            
            Memory functionality enables continuity across multiple interactions.
            """,
            
            "text_generation.txt": """
            Text generation is suitable for:
            - Creating structured datasets with fields and records
            - Generating fictional profiles, stories, or narratives
            - Creating sample data for testing or demonstrations
            - Generating lists of items matching specific criteria
            - Creating formatted text content like articles or documents
            
            To use the text generation system, ask for a specific type of dataset with fields
            or describe the text content you'd like to generate.
            """,
            
            "image_generation.txt": """
            Image generation is suitable for:
            - Creating visual representations of scenes, objects, or concepts
            - Designing artwork in various styles
            - Visualizing landscapes, characters, or abstract concepts
            - Creating visual mockups or prototypes
            - Generating visual examples of described scenarios
            
            To use the image generation system, describe the visual content you'd like to create.
            """,
            
            "memory_system.txt": """
            The memory system keeps track of previously generated content and allows you to:
            - Ask questions about content generated in previous interactions
            - Refer to specific datasets or images by their memory ID
            - Get summaries of previously generated content
            - Request modifications or extensions to previously generated data
            
            Example memory-related queries:
            - "What datasets did I generate recently?"
            - "Tell me about the superhero dataset I created"
            - "Show me details about memory ID text_12345abc"
            - "What can you tell me about the landscape images we generated earlier?"
            """
        }
        
        documents = []
        ids = []
        embeddings = []
        metadata = []
        
        # Save knowledge to files and prepare for embedding
        for i, (filename, content) in enumerate(knowledge_content.items()):
            with open(f"./knowledge_docs/{filename}", "w") as f:
                f.write(content)
            
            doc_id = f"doc_{i}"
            embedding = self.embed_model.encode(content).tolist()
            
            documents.append(content)
            ids.append(doc_id)
            embeddings.append(embedding)
            metadata.append({"source": filename})
        
        self.knowledge_collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )
        
        print("Knowledge base initialized with system documentation.")
    
    def identify_query_type(self, query: str) -> Dict:
        """Identify if a query is about generation or memory retrieval"""
        try:
            # First, try to detect via LLM
            prompt = f"""
            Analyze this user query and determine if it's:
            1. Requesting new text/data generation
            2. Requesting new image generation
            3. Asking about previously generated content (memory)
            
            Return JSON with this format:
            {{
                "query_type": "generation" or "memory",
                "generation_type": "text" or "image" or null,
                "memory_focus": "text" or "image" or "general" or null,
                "confidence": 0-1 value,
                "explanation": "brief explanation"
            }}
            
            User Query: {query}
            """
            
            response = self.llm.generate_content(prompt)
            try:
                analysis = json.loads(response.text)
                return analysis
            except:
                # If parsing fails, use keyword matching as fallback
                pass
        except Exception as e:
            print(f"Error in query type identification: {str(e)}")
        
        # Fallback to keyword matching
        memory_keywords = ["previous", "generated", "created", "last time", "remember", 
                          "recall", "show me", "tell me about", "what was", "memory"]
        
        text_keywords = ["data", "list", "table", "dataset", "text", "write", "create content"]
        image_keywords = ["image", "picture", "photo", "draw", "visual", "art", "illustration"]
        
        query_lower = query.lower()
        
        # Count keywords in each category
        memory_count = sum(1 for kw in memory_keywords if kw in query_lower)
        text_count = sum(1 for kw in text_keywords if kw in query_lower)
        image_count = sum(1 for kw in image_keywords if kw in query_lower)
        
        # Determine query type based on keyword counts
        if memory_count > max(text_count, image_count):
            memory_focus = "general"
            if text_count > image_count:
                memory_focus = "text"
            elif image_count > text_count:
                memory_focus = "image"
            
            return {
                "query_type": "memory",
                "generation_type": None,
                "memory_focus": memory_focus,
                "confidence": 0.7,
                "explanation": "Query contains memory-related keywords."
            }
        elif text_count > image_count:
            return {
                "query_type": "generation",
                "generation_type": "text",
                "memory_focus": None,
                "confidence": 0.7,
                "explanation": "Query contains text generation keywords."
            }
        elif image_count > text_count:
            return {
                "query_type": "generation",
                "generation_type": "image",
                "memory_focus": None,
                "confidence": 0.7,
                "explanation": "Query contains image generation keywords."
            }
        else:
            return {
                "query_type": "unknown",
                "generation_type": None,
                "memory_focus": None,
                "confidence": 0.5,
                "explanation": "Unable to determine query type with confidence."
            }
    
    def query_knowledge_base(self, query: str) -> str:
        """Query the static knowledge base"""
        if not self.embed_model or not self.db:
            return "Knowledge base unavailable."
        
        try:
            query_embedding = self.embed_model.encode(query).tolist()
            
            results = self.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
            
            if results and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
                context = "\n\n".join(results['documents'][0])
                return context
            else:
                return "No relevant information found in knowledge base."
        except Exception as e:
            print(f"Error querying knowledge base: {str(e)}")
            return "Error accessing knowledge base."
    
    def search_memory(self, query: str) -> Dict:
        """Search the memory system for relevant content"""
        if not self.memory_system:
            return {
                "success": False,
                "reason": "Memory system unavailable",
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
                memory_data = self.memory_system.get_data_by_id(memory_id)
                if memory_data:
                    return {
                        "success": True,
                        "query_type": "specific_id",
                        "memory_id": memory_id,
                        "results": [memory_data],
                        "summary": self.memory_system.get_human_readable_memory_summary(memory_data)
                    }
                else:
                    return {
                        "success": False,
                        "query_type": "specific_id",
                        "reason": f"Memory with ID {memory_id} not found",
                        "results": []
                    }
            
            # Otherwise, search semantically
            results = self.memory_system.search_memory(query)
            
            if results and len(results) > 0:
                # Get full data for top result
                top_result_id = results[0]["id"]
                top_result_data = self.memory_system.get_data_by_id(top_result_id)
                
                if top_result_data:
                    memory_summary = self.memory_system.get_human_readable_memory_summary(top_result_data)
                    
                    return {
                        "success": True,
                        "query_type": "semantic_search",
                        "results": results,
                        "top_result": top_result_data,
                        "summary": memory_summary
                    }
            
            # If we get here, no relevant results
            recent_data = self.memory_system.get_recent_data(limit=3)
            
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
        memory_result = self.search_memory(query)
        
        if not memory_result["success"]:
            # No memory matches
            if "recent_items" in memory_result and memory_result["recent_items"]:
                recent = memory_result["recent_items"]
                response = f"I couldn't find memories specifically matching your query, but here are your {len(recent)} most recent generations:\n\n"
                
                for i, item in enumerate(recent, 1):
                    response += f"{i}. {item.get('type', 'Unknown').title()} data: {item.get('description', 'No description')} (ID: {item.get('id', 'unknown')})\n"
                
                response += "\nYou can ask about any of these specifically by mentioning their ID."
                return response
            else:
                return f"I don't have any stored memories to answer your query. {memory_result.get('reason', '')}"
        
        # We found memory matches
        if memory_result["query_type"] == "specific_id":
            # User asked about specific memory ID
            memory_data = memory_result["results"][0]
            summary = memory_result["summary"]
            
            return f"Here's the information about memory {memory_data.get('id', 'unknown')}:\n\n{summary}"
        
        elif memory_result["query_type"] == "semantic_search":
            # Semantic search results
            top_result = memory_result["top_result"]
            summary = memory_result["summary"]
            memory_type = top_result.get("type", "unknown").title()
            
            response = f"I found a relevant {memory_type} generation that matches your query:\n\n{summary}\n\n"
            
            # Add reference to ID for follow-up
            if "id" in top_result:
                response += f"You can refer to this content in future queries using ID: {top_result['id']}"
            
            return response
    
    def enhance_generation_query(self, query: str, query_type: str) -> Dict:
        """Enhance a generation query with relevant knowledge and memory context"""
        knowledge_context = self.query_knowledge_base(query)
        
        # For generation queries, add relevant memory content if helpful
        memory_search = self.search_memory(query)
        memory_context = ""
        
        if memory_search["success"]:
            # Found relevant memory that might be useful as context
            top_result = memory_search["top_result"]
            memory_type = top_result.get("type", "unknown")
            
            # Only add memory context if it's relevant to the current generation type
            if (query_type == "text" and memory_type == "text") or (query_type == "image" and memory_type == "image"):
                if "description" in top_result:
                    memory_context = f"Previous {memory_type} generation: {top_result['description']}"
        
        return {
            "enhanced_query": query,
            "knowledge_context": knowledge_context,
            "memory_context": memory_context
        }
