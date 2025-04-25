from typing import TypedDict, Literal, List, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os
from dotenv import load_dotenv
import text_gen
import image_gen
import json
from datetime import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)
router_model = genai.GenerativeModel('gemini-2.0-flash')

# Enhanced RouterState to include context memory
class RouterState(TypedDict):
    user_query: str
    agent_type: Literal["text", "image", "unknown"]
    routing_reason: str
    output: str
    context_history: List[Dict[str, Any]]  # Store previous interactions

def analyze_query_intent(state: RouterState):
    """Analyze the user's query to determine if it's text or image generation"""
    user_query = state["user_query"]
    context_history = state.get("context_history", [])
    
    print("\n🧠 Analyzing query intent...")
    
    # Include context history in the prompt if available
    context_info = ""
    if context_history:
        context_info = "Previous interactions:\n"
        for i, interaction in enumerate(context_history[-3:]):  # Only use most recent 3 interactions
            timestamp = interaction.get("timestamp", "")
            query = interaction.get("query", "")
            agent = interaction.get("agent_type", "")
            summary = interaction.get("summary", "")
            context_info += f"{i+1}. [{timestamp}] Query: '{query}' (Agent: {agent})\n   Summary: {summary}\n"
    
    # Use the router model to classify the query
    try:
        prompt = f"""
        {context_info}
        
        Analyze the following user query and determine if it's asking for:
        1. Text data generation (like creating datasets, structured data, text content)
        2. Image generation (like creating visualizations, artwork, diagrams)
        
        Consider if the query references previously generated content or asks for a modification
        of previous outputs. If so, route to the same agent type as before.
        
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
            # Check if query asks about previous context
            elif any(word in user_query.lower() for word in ["previous", "before", "last time", "earlier"]):
                # Use the agent type from the most recent interaction if available
                if context_history:
                    agent_type = context_history[-1].get("agent_type", "unknown")
                    explanation = f"Query references previous interaction (using {agent_type} agent)"
                else:
                    agent_type = "unknown"
                    explanation = "Query seems to reference previous context but no history available"
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

def route_to_text_agent(state: RouterState):
    """Send the query to the text data generation agent"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    context_history = state.get("context_history", [])
    
    print(f"\n📝 Routing to TEXT GENERATION agent...")
    print(f"Reason: {reason}")
    
    try:
        # Update text_gen.agent to accept context history
        result = text_gen.agent.invoke({
            "user_query": user_query, 
            "context_history": context_history
        })
        
        output = f"""
        ====== TEXT GENERATION RESULT ======
        
        ROUTING REASON: {reason}
        
        {result["output"]}
        """
        
        # Create a summary of this interaction for context history
        summary = result.get("summary", "Generated text data")
        
        # Update context history
        new_interaction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": user_query,
            "agent_type": "text",
            "output": result["output"],
            "summary": summary
        }
        
        # Add new interaction to context history
        updated_history = context_history + [new_interaction]
        if len(updated_history) > 10:  # Keep only the last 10 interactions
            updated_history = updated_history[-10:]
        
        return {
            "output": output,
            "context_history": updated_history
        }
    except Exception as e:
        return {
            "output": f"Error while processing with text agent: {str(e)}\n\nOriginal routing reason: {reason}",
            "context_history": context_history  # Keep context unchanged on error
        }

def route_to_image_agent(state: RouterState):
    """Send the query to the image generation agent"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    context_history = state.get("context_history", [])
    
    print(f"\n🖼️ Routing to IMAGE GENERATION agent...")
    print(f"Reason: {reason}")
    
    try:
        # Update image_gen.agent to accept context history
        result = image_gen.agent.invoke({
            "user_query": user_query,
            "context_history": context_history
        })
        
        output = f"""
        ====== IMAGE GENERATION RESULT ======
        
        ROUTING REASON: {reason}
        
        {result["output"]}
        """
        
        # Create a summary of this interaction for context history
        summary = result.get("summary", "Generated image visualization")
        
        # Update context history
        new_interaction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": user_query,
            "agent_type": "image",
            "output": result["output"],
            "summary": summary
        }
        
        # Add new interaction to context history
        updated_history = context_history + [new_interaction]
        if len(updated_history) > 10:  # Keep only the last 10 interactions
            updated_history = updated_history[-10:]
        
        return {
            "output": output,
            "context_history": updated_history
        }
    except Exception as e:
        return {
            "output": f"Error while processing with image agent: {str(e)}\n\nOriginal routing reason: {reason}",
            "context_history": context_history  # Keep context unchanged on error
        }

def handle_unknown_query(state: RouterState):
    """Handle queries that couldn't be clearly classified"""
    user_query = state["user_query"]
    reason = state["routing_reason"]
    context_history = state.get("context_history", [])
    
    print(f"\n❓ Handling UNKNOWN query type...")
    print(f"Reason: {reason}")
    
    try:
        # Include context history in the prompt if available
        context_info = ""
        if context_history:
            context_info = "Previous interactions:\n"
            for i, interaction in enumerate(context_history[-3:]):  # Only use most recent 3 interactions
                timestamp = interaction.get("timestamp", "")
                query = interaction.get("query", "")
                agent = interaction.get("agent_type", "")
                summary = interaction.get("summary", "")
                context_info += f"{i+1}. [{timestamp}] Query: '{query}' (Agent: {agent})\n   Summary: {summary}\n"
        
        # Default to using Gemini for general queries
        prompt = f"""
        {context_info}
        
        The user's query could not be clearly identified as text or image generation.
        Please respond helpfully to this query, taking into account any relevant previous interactions: 
        
        {user_query}
        
        Include details about the system's capabilities:
        - Can generate text datasets with fields and records
        - Can create image visualizations and artwork
        - Can reference previously generated content
        """
        
        response = router_model.generate_content(prompt)
        
        output = f"""
        ====== GENERAL RESPONSE ======
        
        ROUTING REASON: {reason}
        
        {response.text}
        
        TIP: For better results, try specifying if you want to generate text data or create images.
        """
        
        # Create a summary of this interaction for context history
        summary = "General response to ambiguous query"
        
        # Update context history
        new_interaction = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": user_query,
            "agent_type": "unknown",
            "output": response.text,
            "summary": summary
        }
        
        # Add new interaction to context history
        updated_history = context_history + [new_interaction]
        if len(updated_history) > 10:  # Keep only the last 10 interactions
            updated_history = updated_history[-10:]
        
        return {
            "output": output,
            "context_history": updated_history
        }
    except Exception as e:
        return {
            "output": f"Error while processing with general handler: {str(e)}\n\nOriginal routing reason: {reason}",
            "context_history": context_history  # Keep context unchanged on error
        }

def route_based_on_type(state: RouterState):
    """Route to the appropriate agent based on query type"""
    agent_type = state["agent_type"]
    
    if agent_type == "text":
        return "TEXT"
    elif agent_type == "image":
        return "IMAGE"
    else:
        return "UNKNOWN"

# Define the workflow
workflow = StateGraph(RouterState)

# Add all nodes
workflow.add_node("analyze_intent", analyze_query_intent)
workflow.add_node("route_to_text", route_to_text_agent)
workflow.add_node("route_to_image", route_to_image_agent)
workflow.add_node("handle_unknown", handle_unknown_query)

# Set up the workflow edges
workflow.add_conditional_edges(
    "analyze_intent",
    route_based_on_type,
    {
        "TEXT": "route_to_text",
        "IMAGE": "route_to_image",
        "UNKNOWN": "handle_unknown"
    }
)

# Connect all outputs to END
workflow.add_edge("route_to_text", END)
workflow.add_edge("route_to_image", END)
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
    print("- System remembers previous interactions for context-aware responses")
    
    # Initialize context history
    context_history = []
    
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
            
            # Pass the existing context history to the agent
            result = agent.invoke({
                "user_query": user_query,
                "context_history": context_history
            })
            
            # Update the context history for the next interaction
            context_history = result.get("context_history", context_history)
            
            print("\n" + result["output"])
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different request.")