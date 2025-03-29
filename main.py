from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
import chromadb
import image_gen
import text_gen

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)
router_model = genai.GenerativeModel('gemini-2.0-flash')

# Directory setup
os.makedirs("./knowledge_docs", exist_ok=True)
os.makedirs("./chroma_db", exist_ok=True)

# RAG System using Sentence Transformers and ChromaDB
class RAGSystem:
    def __init__(self):
        try:
            print("Loading embedding model...")
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            self.db = chromadb.PersistentClient(path="./chroma_db")
            self.chroma_collection = self.db.get_or_create_collection("knowledge_base")
            
            # Load documents (only needs to be done once)
            if len(self.chroma_collection.get()['ids']) == 0:
                self._initialize_knowledge_base()
            
            print("RAG system initialized successfully.")
        except Exception as e:
            raise Exception(f"Failed to initialize RAG system: {str(e)}")
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base with sample documents"""
        print("Initializing knowledge base...")
        
        # Sample content
        sample_content = {
            "generation_types.txt": """
            The system supports two types of generation:
            1. Text Generation: For creating datasets, lists, stories, and textual content
            2. Image Generation: For creating visual artwork, designs, and illustrations
            
            Text generation keywords: data, list, table, generate text, create content
            Image generation keywords: image, picture, photo, draw, visual, generate a
            """,
            "capabilities.txt": """
            System capabilities include:
            - Generating realistic dataset examples
            - Creating fictional character profiles
            - Producing landscape images
            - Designing abstract art
            - Generating technical diagrams
            """,
            "text_generation.txt": """
            Text generation is suitable for:
            - Creating structured datasets with fields and records
            - Generating fictional profiles, stories, or narratives
            - Creating sample data for testing or demonstrations
            - Generating lists of items matching specific criteria
            - Creating formatted text content like articles or documents
            """,
            "image_generation.txt": """
            Image generation is suitable for:
            - Creating visual representations of scenes, objects, or concepts
            - Designing artwork in various styles
            - Visualizing landscapes, characters, or abstract concepts
            - Creating visual mockups or prototypes
            - Generating visual examples of described scenarios
            """
        }
        
        # Save sample documents
        documents = []
        ids = []
        embeddings = []
        metadata = []
        
        for i, (filename, content) in enumerate(sample_content.items()):
            # Save to file for reference
            with open(f"./knowledge_docs/{filename}", "w") as f:
                f.write(content)
            
            # Add to ChromaDB
            doc_id = f"doc_{i}"
            embedding = self.embed_model.encode(content).tolist()
            
            documents.append(content)
            ids.append(doc_id)
            embeddings.append(embedding)
            metadata.append({"source": filename})
        
        # Add all documents to ChromaDB
        self.chroma_collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )
        
        print("Knowledge base initialized successfully.")

    def query(self, question: str) -> str:
        """Query the knowledge base"""
        # Embed the query
        query_embedding = self.embed_model.encode(question).tolist()
        
        # Search for similar documents
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        
        if results and len(results['documents']) > 0:
            # Combine the top results
            context = "\n\n".join(results['documents'][0])
            return context
        else:
            return "No relevant information found."

class RouterState(TypedDict):
    user_query: str
    agent_type: Literal["text", "image", "unknown"]
    routing_reason: str
    rag_context: Annotated[str, "Relevant information retrieved from knowledge base"]
    output: str

# Initialize RAG system
try:
    rag_system = RAGSystem()
    rag_enabled = True
except Exception as e:
    print(f"Warning: Failed to initialize RAG system: {str(e)}")
    print("Continuing without RAG capabilities...")
    rag_enabled = False

def enhance_with_rag(state: RouterState):
    """Enhance the query with RAG knowledge"""
    if not rag_enabled:
        return {
            "rag_context": "RAG system is disabled. Processing without knowledge enhancement."
        }
    
    print("\n🔍 Consulting knowledge base...")
    
    try:
        # Get relevant context from RAG system
        rag_query = (
            f"Based on the following user request, provide information about which "
            f"generation type (text or image) is most appropriate and why: {state['user_query']}"
        )
        rag_context = rag_system.query(rag_query)
        
        print(f"📚 Retrieved context: {rag_context[:100]}...")
        
        return {
            "rag_context": rag_context
        }
    except Exception as e:
        print(f"⚠️ RAG query failed: {str(e)}")
        return {
            "rag_context": f"Failed to retrieve context: {str(e)}"
        }

def determine_agent_type(state: RouterState):
    """Analyze the enhanced query to determine which agent should handle it"""
    user_query = state["user_query"]
    rag_context = state["rag_context"] if rag_enabled else ""
    
    print("\n🤔 Analyzing request...")
    
    # Basic prompt if RAG is not available
    basic_prompt = """
    Analyze this user query and determine if it's requesting:
    1. Text/data generation - respond with 'text'
    2. Image generation - respond with 'image'
    3. If unclear - respond with 'unknown'
    
    Return ONLY the classification without explanation or additional text.
    
    User Query: 
    """
    
    # Enhanced prompt with RAG
    rag_prompt = """
    Analyze this user query with the retrieved system knowledge:
    
    USER QUERY: {query}
    
    SYSTEM KNOWLEDGE: {context}
    
    Determine if this request is for:
    1. Text/data generation - respond with 'text'
    2. Image generation - respond with 'image'
    3. If unclear - respond with 'unknown'
    
    Return ONLY the classification without explanation or additional text.
    """
    
    try:
        if rag_enabled:
            prompt = rag_prompt.format(query=user_query, context=rag_context)
        else:
            prompt = basic_prompt + user_query
            
        response = router_model.generate_content(prompt)
        agent_type = response.text.strip().lower()
        
        # Validate agent_type
        if agent_type not in ["text", "image", "unknown"]:
            print(f"⚠️ Invalid classification: '{agent_type}', defaulting to 'unknown'")
            agent_type = "unknown"
            reason = "Classification result was invalid"
        else:
            # Generate a reason for the classification separately
            reason_prompt = f"""
            Explain briefly why this query: "{user_query}" 
            should be classified as {agent_type} generation.
            Keep it to one sentence.
            """
            reason_response = router_model.generate_content(reason_prompt)
            reason = reason_response.text.strip()
        
        print(f"🧠 Classification: {agent_type.upper()} - {reason[:50]}...")
        
        return {
            "agent_type": agent_type,
            "routing_reason": reason
        }
    except Exception as e:
        print(f"⚠️ Classification error: {str(e)}")
        return {
            "agent_type": "unknown",
            "routing_reason": f"Error during classification: {str(e)}"
        }

def route_to_text_agent(state: RouterState):
    """Send the query to the text data generation agent"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    
    print(f"\n📝 Routing to TEXT GENERATION agent...")
    print(f"Reason: {reason}")
    
    try:
        # Enhance the query with the RAG context if available
        enhanced_query = user_query
        if rag_enabled and state["rag_context"]:
            enhanced_query = f"{user_query}\n\nAdditional context: {state['rag_context']}"
        
        # Invoke the text_gen agent directly from the imported module
        result = text_gen.agent.invoke({"user_query": enhanced_query})
        
        output = f"""
        ====== TEXT GENERATION RESULT ======
        
        ROUTING REASON: {reason}
        
        {result["output"]}
        """
        
        return {"output": output}
    except Exception as e:
        return {
            "output": f"Error while processing with text agent: {str(e)}\n\nOriginal routing reason: {reason}"
        }

def route_to_image_agent(state: RouterState):
    """Send the query to the image generation agent"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    
    print(f"\n🖼️ Routing to IMAGE GENERATION agent...")
    print(f"Reason: {reason}")
    
    try:
        # Enhance the query with the RAG context if available
        enhanced_query = user_query
        if rag_enabled and state["rag_context"]:
            enhanced_query = f"{user_query}\n\nAdditional context: {state['rag_context']}"
        
        # Invoke the image_gen agent directly from the imported module
        result = image_gen.agent.invoke({"user_query": enhanced_query})
        
        output = f"""
        ====== IMAGE GENERATION RESULT ======
        
        ROUTING REASON: {reason}
        
        {result["output"]}
        """
        
        return {"output": output}
    except Exception as e:
        return {
            "output": f"Error while processing with image agent: {str(e)}\n\nOriginal routing reason: {reason}"
        }

def handle_unknown_query(state: RouterState):
    """Handle queries that couldn't be classified"""
    query = state["user_query"]
    reason = state["routing_reason"]
    
    print("\n❓ Query type is UNKNOWN")
    
    prompt = f"""
    I couldn't determine if you want text or image generation from your request: "{query}"
    
    Please try again with a clearer request, such as:
    
    For TEXT generation:
    - "Generate a list of 10 science fiction book ideas"
    - "Write a marketing email for a new fitness product"
    - "Create a dataset of fictional customer profiles"
    
    For IMAGE generation:
    - "Create an image of a sunset over mountains"
    - "Generate a portrait of a cyberpunk character"
    - "Design a logo for a coffee shop"
    """
    
    response = router_model.generate_content(prompt)
    
    output = f"""
    ====== UNABLE TO DETERMINE REQUEST TYPE ======
    
    {response.text.strip()}
    
    Original routing reason: {reason}
    """
    
    return {"output": output}

# Build the enhanced router workflow
workflow = StateGraph(RouterState)

# Add nodes
workflow.add_node("enhance_with_rag", enhance_with_rag)
workflow.add_node("determine_agent", determine_agent_type)
workflow.add_node("text_gen", route_to_text_agent)
workflow.add_node("image_gen", route_to_image_agent)
workflow.add_node("unknown", handle_unknown_query)

# Add edges with RAG enhancement first
workflow.add_edge("enhance_with_rag", "determine_agent")

# Add conditional branching based on agent type
workflow.add_conditional_edges(
    "determine_agent",
    lambda state: state["agent_type"],
    {
        "text": "text_gen",
        "image": "image_gen",
        "unknown": "unknown"
    }
)

# Connect all endpoints
workflow.add_edge("text_gen", END)
workflow.add_edge("image_gen", END)
workflow.add_edge("unknown", END)

# Set entry point
workflow.set_entry_point("enhance_with_rag")
router_agent = workflow.compile()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌟 RAG-Enhanced Generation Router 🌟")
    print("="*60)
    print("This system intelligently routes your requests to")
    print("the most appropriate generation agent (text or image).")
    if rag_enabled:
        print("Knowledge enhancement is ENABLED ✓")
    else:
        print("Knowledge enhancement is DISABLED ✗")
    print("="*60)
    
    while True:
        try:
            user_query = input("\n🔍 What would you like to generate? (or 'exit' to quit): ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye! Thank you for using the system.")
                break
                
            if not user_query:
                continue
                
            print("\n🚀 Processing your request...")
            
            # Initialize default state
            initial_state = {
                "user_query": user_query,
                "agent_type": "unknown",
                "routing_reason": "",
                "output": "",
                "rag_context": ""
            }
            
            # Invoke the router agent
            result = router_agent.invoke(initial_state)
            
            # Display results
            print("\n" + "="*60)
            print("📊 ROUTING DETAILS:")
            print(f"• Agent Type: {result['agent_type'].upper()}")
            print(f"• Reason: {result['routing_reason']}")
            print("\n📋 RESULT:")
            print(result["output"])
            print("="*60)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different query.")
