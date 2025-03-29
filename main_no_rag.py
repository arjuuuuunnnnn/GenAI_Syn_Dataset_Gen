from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os
from dotenv import load_dotenv
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

class RouterState(TypedDict):
    user_query: str
    query_type: Annotated[str, "Determined query type: 'image' or 'text'"]
    response: Annotated[str, "Final response to return to the user"]

def classify_query(state: RouterState):
    """Determine if the query is for image generation or text data generation"""
    user_query = state["user_query"]
    print("\n🧠 Analyzing query type...")
    
    prompt = f"""
    Analyze the following user query and determine if it's requesting:
    1. IMAGE generation (pictures, visuals, artwork, etc.)
    2. TEXT DATA generation (datasets, records, information, etc.)
    
    Return ONLY "image" or "text" without any explanation.
    
    User Query: {user_query}
    """
    
    response = router_model.generate_content(prompt)
    query_type = response.text.strip().lower()
    
    # Ensure we get a valid response
    if query_type not in ["image", "text"]:
        # Default to text if classification is unclear
        print(f"⚠️ Classification unclear: '{query_type}', defaulting to text")
        query_type = "text"
    
    print(f"📊 Query classified as: {query_type.upper()}")
    return {"query_type": query_type}

def route_to_image_agent(state: RouterState):
    """Send the query to the image generation agent"""
    user_query = state["user_query"]
    print("\n🖼️ Routing to image generation agent...")
    
    # Invoke the image generation agent
    result = image_gen.agent.invoke({"user_query": user_query})
    
    return {"response": f"IMAGE GENERATION RESULT:\n{result['output']}"}

def route_to_text_agent(state: RouterState):
    """Send the query to the text data generation agent"""
    user_query = state["user_query"]
    print("\n📝 Routing to text data generation agent...")
    
    # Invoke the text generation agent
    result = text_gen.agent.invoke({"user_query": user_query})
    
    return {"response": f"TEXT DATA GENERATION RESULT:\n{result['output']}"}

# Build the router workflow
def should_route_to_image(state: RouterState):
    """Conditional routing based on query type"""
    return state["query_type"] == "image"

workflow = StateGraph(RouterState)
workflow.add_node("classify", classify_query)
workflow.add_node("image_agent", route_to_image_agent)
workflow.add_node("text_agent", route_to_text_agent)

# Add conditional branching
workflow.add_conditional_edges(
    "classify",
    should_route_to_image,
    {
        True: "image_agent",
        False: "text_agent"
    }
)

workflow.add_edge("image_agent", END)
workflow.add_edge("text_agent", END)

workflow.set_entry_point("classify")
router_agent = workflow.compile()

if __name__ == "__main__":
    while True:
        try:
            print("\n" + "="*50)
            print("🤖 Multi-Agent System - Enter 'exit' to quit")
            print("="*50)
            query = input("What would you like to generate? ")
            
            if query.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break
                
            print("\n🚀 Processing your request...")
            result = router_agent.invoke({"user_query": query})
            
            print("\n" + "="*50)
            print(result["response"])
            print("="*50)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different query.")
