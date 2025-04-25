# Add this code to main2.py

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os
from dotenv import load_dotenv
import text_gen
import image_gen
from data_memory import GeneratedDataMemory

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)
router_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize memory system
try:
    memory_system = GeneratedDataMemory()
except Exception as e:
    print(f"Warning: Memory system initialization failed: {str(e)}")
    memory_system = None

class RouterState(TypedDict):
    user_query: str
    agent_type: Literal["text", "image", "unknown", "memory"]
    routing_reason: str
    memory_context: str
    output: str

def analyze_query_intent(state: RouterState):
    """Analyze the user's query to determine if it's text or image generation, or memory retrieval"""
    user_query = state["user_query"]
    
    print("\n🧠 Analyzing query intent...")
    
    # Check if the query starts with "please" for memory retrieval
    if user_query.lower().strip().startswith("please"):
        print("📚 Detected memory retrieval request...")
        return {
            "agent_type": "memory",
            "routing_reason": "Query starts with 'please' - retrieving context from memory."
        }
    
    # Use the router model to classify the query (existing code)
    try:
        prompt = f"""
        Analyze the following user query and determine if it's asking for:
        1. Text data generation (like creating datasets, structured data, text content)
        2. Image generation (like creating visualizations, artwork, diagrams)
        
        Return your analysis in JSON format with two fields:
        - "agent_type": either "text", "image", or "unknown"
        - "explanation": brief explanation of your classification
        
        User query: {user_query}
        """
        
        response = router_model.generate_content(prompt)
        
        # Parse the response to get JSON
        response_text = response.text
        # Find JSON content between triple backticks if present
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_content = response_text.split("```")[1].strip()
        else:
            json_content = response_text.strip()
            
        try:
            analysis = eval(json_content)
            agent_type = analysis.get("agent_type", "unknown")
            explanation = analysis.get("explanation", "No explanation provided")
        except:
            # Fallback if JSON parsing fails
            if "image" in user_query.lower():
                agent_type = "image"
                explanation = "Query contains 'image' keyword"
            elif any(word in user_query.lower() for word in ["data", "text", "dataset"]):
                agent_type = "text"
                explanation = "Query contains text/data generation keywords"
            else:
                agent_type = "unknown"
                explanation = "Could not classify query intent reliably"
        
        print(f"📊 Query intent: {agent_type.upper()} - {explanation}")
        
        return {
            "agent_type": agent_type,
            "routing_reason": explanation
        }
        
    except Exception as e:
        print(f"⚠️ Error analyzing query intent: {str(e)}")
        return {
            "agent_type": "unknown",
            "routing_reason": f"Error analyzing query intent: {str(e)}"
        }

def process_memory_retrieval(state: RouterState):
    """Search memory system for relevant context and process with Gemini"""
    user_query = state["user_query"]
    print("\n🔍 Retrieving relevant memory context...")
    
    memory_context = ""
    search_results = []
    
    try:
        if not memory_system:
            memory_context = "Memory system not available."
        else:
            # Remove "please" from the query for better search results
            search_query = user_query.lower().replace("please", "", 1).strip()
            
            # Search memory for relevant results - using the search_memory method directly
            # This method already handles the embedding internally
            search_results = memory_system.search_memory(search_query, limit=3)
            
            if not search_results:
                memory_context = "No relevant memory entries found."
            else:
                memory_context = "Found relevant memory entries:\n\n"
                
                for i, result in enumerate(search_results):
                    # Get detailed data for each result
                    memory_data = memory_system.get_data_by_id(result["id"])
                    if memory_data:
                        summary = memory_system.get_human_readable_memory_summary(memory_data)
                        memory_context += f"--- Memory Entry {i+1} ---\n{summary}\n\n"
        
        print(f"Retrieved {len(search_results) if search_results else 0} memory entries")
        
        # Process with Gemini using the memory context
        prompt = f"""
        The user has asked:
        "{user_query}"
        
        Here is the relevant context from previous interactions:
        {memory_context}
        
        Based on this context, please provide a helpful response to the user's query.
        Include specific details from the memory entries when relevant.
        If the memory doesn't contain relevant information, acknowledge this and provide the best response possible.
        """
        
        response = router_model.generate_content(prompt)
        
        output = f"""
        ====== MEMORY-ASSISTED RESPONSE ======
        
        {response.text}
        """
        
        return {"memory_context": memory_context, "output": output}
        
    except Exception as e:
        error_message = f"Error processing memory request: {str(e)}"
        print(f"⚠️ {error_message}")
        return {
            "memory_context": error_message,
            "output": f"I encountered an error while retrieving memory: {str(e)}\n\nPlease try again with a different query."
        }

def route_to_text_agent(state: RouterState):
    """Send the query to the text data generation agent"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    
    print(f"\n📝 Routing to TEXT GENERATION agent...")
    print(f"Reason: {reason}")
    
    try:
        result = text_gen.agent.invoke({"user_query": user_query})
        
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
        result = image_gen.agent.invoke({"user_query": user_query})
        
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
        The user's query could not be clearly identified as text or image generation.
        Please respond helpfully to this query: {user_query}
        
        Include details about the system's capabilities:
        - Can generate text datasets with fields and records
        - Can create image visualizations and artwork
        """)
        
        output = f"""
        ====== GENERAL RESPONSE ======
        
        ROUTING REASON: {reason}
        
        {response.text}
        
        TIP: For better results, try specifying if you want to generate text data or create images.
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
workflow.add_node("route_to_text", route_to_text_agent)
workflow.add_node("route_to_image", route_to_image_agent)
workflow.add_node("process_memory", process_memory_retrieval)
workflow.add_node("handle_unknown", handle_unknown_query)

# Set up the workflow edges
workflow.add_conditional_edges(
    "analyze_intent",
    route_based_on_type,
    {
        "TEXT": "route_to_text",
        "IMAGE": "route_to_image",
        "MEMORY": "process_memory",
        "UNKNOWN": "handle_unknown"
    }
)

# Connect all outputs to END
workflow.add_edge("route_to_text", END)
workflow.add_edge("route_to_image", END)
workflow.add_edge("process_memory", END)
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
    print("- Memory retrieval (start query with 'please')")
    
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
