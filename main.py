from typing import TypedDict, Literal, Annotated, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
import torch
from data_memory import GeneratedDataMemory
from memory_rag import MemoryAwareRAG
import image_gen
import text_gen

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)
router_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize our enhanced RAG system
try:
    rag_system = MemoryAwareRAG()
    rag_enabled = True
except Exception as e:
    print(f"Warning: Failed to initialize RAG system: {str(e)}")
    print("Continuing without RAG capabilities...")
    rag_enabled = False

class RouterState(TypedDict):
    user_query: str
    agent_type: Literal["text", "image", "memory", "unknown"]
    routing_reason: str
    rag_context: Annotated[str, "Relevant information retrieved from knowledge base"]
    memory_response: Annotated[str, "Response from memory system for memory queries"]
    output: str

def analyze_query_intent(state: RouterState):
    """Analyze the user's query to determine if it's generation or memory retrieval"""
    user_query = state["user_query"]
    
    print("\n🧠 Analyzing query intent...")
    
    if not rag_enabled:
        # Without RAG, just guess based on keywords
        return {
            "agent_type": "unknown",
            "routing_reason": "RAG system disabled, unable to determine query intent reliably"
        }
    
    try:
        # Use the enhanced RAG system to identify query type
        query_analysis = rag_system.identify_query_type(user_query)
        
        query_type = query_analysis.get("query_type", "unknown")
        generation_type = query_analysis.get("generation_type")
        memory_focus = query_analysis.get("memory_focus")
        explanation = query_analysis.get("explanation", "No explanation provided")
        
        if query_type == "memory":
            agent_type = "memory"
            reason = f"Query identified as memory retrieval: {explanation}"
        elif query_type == "generation":
            if generation_type == "text":
                agent_type = "text"
                reason = f"Query identified as text generation: {explanation}"
            elif generation_type == "image":
                agent_type = "image"
                reason = f"Query identified as image generation: {explanation}"
            else:
                agent_type = "unknown"
                reason = f"Generation type could not be determined: {explanation}"
        else:
            agent_type = "unknown"
            reason = f"Query intent unclear: {explanation}"
        
        print(f"📊 Query intent: {agent_type.upper()} - {reason}")
        
        return {
            "agent_type": agent_type,
            "routing_reason": reason
        }
        
    except Exception as e:
        print(f"⚠️ Error analyzing query intent: {str(e)}")
        return {
            "agent_type": "unknown",
            "routing_reason": f"Error analyzing query intent: {str(e)}"
        }

def enhance_with_rag(state: RouterState):
    """Enhance the query with RAG knowledge"""
    if not rag_enabled:
        return {
            "rag_context": "RAG system is disabled. Processing without knowledge enhancement."
        }
    
    print("\n🔍 Enhancing query with knowledge...")
    user_query = state["user_query"]
    agent_type = state["agent_type"]
    
    try:
        if agent_type == "memory":
            # For memory queries, we'll handle them separately
            rag_context = "Memory query will be processed by the memory system."
        else:
            # For generation queries, enhance with knowledge base
            enhancement = rag_system.enhance_generation_query(user_query, agent_type)
            rag_context = enhancement.get("knowledge_context", "") 
            if enhancement.get("memory_context"):
                rag_context += f"\n\nRelevant memory context: {enhancement.get('memory_context')}"
        
        print(f"📚 Enhanced with context: {rag_context[:100]}...")
        
        return {
            "rag_context": rag_context
        }
    except Exception as e:
        print(f"⚠️ RAG enhancement failed: {str(e)}")
        return {
            "rag_context": f"Failed to enhance with RAG: {str(e)}"
        }

def handle_memory_query(state: RouterState):
    """Handle queries about previously generated content"""
    user_query = state["user_query"]
    
    print("\n💾 Processing memory query...")
    
    if not rag_enabled:
        response = "Memory system is disabled. I can't access previously generated content."
        return {"memory_response": response, "output": response}
    
    try:
        memory_response = rag_system.answer_memory_query(user_query)
        
        print(f"📋 Memory response: {memory_response[:100]}...")
        
        output = f"""
        ====== MEMORY RETRIEVAL RESULT ======
        
        {memory_response}
        """
        
        return {"memory_response": memory_response, "output": output}
    except Exception as e:
        error_msg = f"Error processing memory query: {str(e)}"
        print(f"⚠️ {error_msg}")
        return {"memory_response": error_msg, "output": error_msg}

def route_to_text_agent(state: RouterState):
    """Send the query to the text data generation agent"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    rag_context = state["rag_context"]
    
    print(f"\n📝 Routing to TEXT GENERATION agent...")
    print(f"Reason: {reason}")
    
    try:
        enhanced_query = user_query
        if rag_enabled and rag_context:
            enhanced_query = f"{user_query}\n\nAdditional context: {rag_context}"
        
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
    rag_context = state["rag_context"]
    
    print(f"\n🖼️ Routing to IMAGE GENERATION agent...")
    print(f"Reason: {reason}")
    
    try:
        enhanced_query = user_query
        if rag_enabled and rag_context:
            enhanced_query = f"{user_query}\n\nAdditional context: {rag_context}"
        
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
    """Handle queries that couldn't be clearly classified"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    
    print(f"\n❓ Handling UNKNOWN query type...")
    print(f"Reason: {reason}")
    
    try:
        # Default to using Gemini for general queries
        response = router_model.generate_content(f"""
        The user's query could not be clearly identified as a generation or memory request.
        Please respond helpfully to this query: {user_query}
        
        Include details about the system's capabilities:
        - Can generate text datasets with fields and records
        - Can create image visualizations and artwork
        - Can remember previously generated content for reference
        """)
        
        output = f"""
        ====== GENERAL RESPONSE ======
        
        ROUTING REASON: {reason}
        
        {response.text}
        
        TIP: For better results, try specifying if you want to generate text data, 
        create images, or retrieve previously generated content.
        """
        
        return {"output": output}
    except Exception as e:
        return {
            "output": f"Error while processing with general handler: {str(e)}\n\nOriginal routing reason: {reason}"
        }

def route_based_on_type(state: RouterState):
    """Route to the appropriate agent based on query type"""
    agent_type = state["agent_type"]
    
    if agent_type == "text":
        return "TEXT"
    elif agent_type == "image":
        return "IMAGE"
    elif agent_type == "memory":
        return "MEMORY"
    else:
        return "UNKNOWN"

# Define the workflow
workflow = StateGraph(RouterState)

# Add all nodes
workflow.add_node("analyze_intent", analyze_query_intent)
workflow.add_node("enhance_context", enhance_with_rag)
workflow.add_node("handle_memory", handle_memory_query)
workflow.add_node("route_to_text", route_to_text_agent)
workflow.add_node("route_to_image", route_to_image_agent)
workflow.add_node("handle_unknown", handle_unknown_query)

# Set up the workflow edges
workflow.add_edge("analyze_intent", "enhance_context")
workflow.add_edge("enhance_context", route_based_on_type)
workflow.add_edge("TEXT", "route_to_text")
workflow.add_edge("IMAGE", "route_to_image")
workflow.add_edge("MEMORY", "handle_memory")
workflow.add_edge("UNKNOWN", "handle_unknown")

# Connect all outputs to END
workflow.add_edge("route_to_text", END)
workflow.add_edge("route_to_image", END)
workflow.add_edge("handle_memory", END)
workflow.add_edge("handle_unknown", END)

# Set the entrypoint
workflow.set_entry_point("analyze_intent")

# Compile the graph into a runnable
agent = workflow.compile()

if __name__ == "__main__":
    print("=== Content Agent Router System ===")
    print("This system routes user queries to specialized agents for:")
    print("- Text data generation")
    print("- Image generation")
    print("- Memory retrieval")
    
    while True:
        try:
            user_query = input("\nEnter your request (or 'exit' to quit): ")
            if user_query.lower() in ["exit", "quit", "q"]:
                print("Exiting system. Goodbye!")
                break
                
            if not user_query.strip():
                print("Please enter a valid query.")
                continue
                
            print("\n🚀 Processing your request...")
            result = agent.invoke({"user_query": user_query})
            
            print("\n" + result["output"])
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different request.")
