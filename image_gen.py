from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import requests
import os
import json
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from datetime import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Configure image generation
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_IMAGES = 4
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to import memory system (optional)
try:
    from data_memory import GeneratedDataMemory
    memory_system = GeneratedDataMemory()
    print("Memory system initialized successfully")
except Exception as e:
    print(f"Warning: Memory system initialization failed: {str(e)}")
    memory_system = None

class ImageAgentState(TypedDict):
    user_query: str
    context_history: List[Dict[str, Any]]
    dataset_description: str
    image_prompts: List[str]
    generated_images: List[dict]
    memory_id: str
    output: str
    summary: str

def analyze_request(state: ImageAgentState):
    """Analyze the user's request with context awareness"""
    user_query = state["user_query"]
    context_history = state.get("context_history", [])
    
    print("\n🔍 Analyzing your request with context...")
    
    # Build context information from relevant history
    context_info = ""
    referenced_memory_ids = []
    
    if context_history:
        relevant_history = context_history[-5:]  # Get the 5 most recent interactions
        context_info = "Previous interactions:\n"
        
        for i, interaction in enumerate(relevant_history):
            query = interaction.get("query", "")
            agent_type = interaction.get("agent_type", "")
            summary = interaction.get("summary", "")
            memory_id = interaction.get("memory_id", None)
            
            context_info += f"{i+1}. Query: '{query}'\n"
            context_info += f"   Agent: {agent_type}\n"
            context_info += f"   Summary: {summary}\n"
            
            if memory_id:
                context_info += f"   Memory ID: {memory_id}\n"
                referenced_memory_ids.append(memory_id)
    
    # Check if query references previous images
    try:
        references_previous = False
        memory_lookup_id = None
        
        # Use Gemini to help determine if the query references previous images
        if context_history:
            reference_prompt = f"""
            Given this user query: "{user_query}"
            
            And these previous interactions:
            {context_info}
            
            Does the user query clearly reference previously generated images or visual content?
            Return ONLY a JSON with two fields:
            {{
                "references_previous": true/false,
                "referenced_memory_id": "memory_id or null if none specified"
            }}
            """
            
            reference_response = gemini_model.generate_content(reference_prompt)
            reference_analysis = extract_json(reference_response.text)
            
            references_previous = reference_analysis.get("references_previous", False)
            memory_lookup_id = reference_analysis.get("referenced_memory_id", None)
            
            # If we have a memory ID referenced, use it. Otherwise, use the most recent one
            if not memory_lookup_id and references_previous and referenced_memory_ids:
                memory_lookup_id = referenced_memory_ids[0]
                
            if references_previous:
                print(f"Query references previous images. Memory ID: {memory_lookup_id}")
        
        # Get details from memory system if available and applicable
        memory_data = None
        if memory_system and memory_lookup_id:
            try:
                memory_data = memory_system.retrieve_data(memory_lookup_id)
                print(f"Retrieved data from memory: {memory_lookup_id}")
            except Exception as e:
                print(f"Error retrieving memory: {str(e)}")
    
        # Generate analysis with Gemini
        analysis_prompt = f"""
        Analyze this image generation request:
        
        USER QUERY: {user_query}
        
        {context_info if context_info else ""}
        
        {f"PREVIOUS IMAGE DATA: {json.dumps(memory_data)}" if memory_data else ""}
        
        Return ONLY a JSON with these fields:
        {{
            "description": "detailed description of the images to create",
            "styles": ["list of artistic styles"],
            "requirements": ["specific requirements"],
            "references_previous": true/false,
            "previous_memory_id": "ID of referenced memory if applicable"
        }}
        """
        
        analysis_response = gemini_model.generate_content(analysis_prompt)
        analysis = extract_json(analysis_response.text)
        
        return {
            "dataset_description": analysis.get("description", user_query),
            "references_previous": analysis.get("references_previous", references_previous),
            "previous_memory_id": analysis.get("previous_memory_id", memory_lookup_id)
        }
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return {
            "dataset_description": user_query,
            "references_previous": False,
            "previous_memory_id": None
        }

def create_prompts(state: ImageAgentState):
    """Generate detailed image prompts with context awareness"""
    description = state["dataset_description"]
    references_previous = state.get("references_previous", False)
    previous_memory_id = state.get("previous_memory_id")
    
    print(f"\n💡 Generating {NUM_IMAGES} detailed prompts...")
    
    # Get previous prompts if applicable
    previous_prompts = []
    if references_previous and memory_system and previous_memory_id:
        try:
            memory_data = memory_system.retrieve_data(previous_memory_id)
            if memory_data and "prompts" in memory_data:
                previous_prompts = memory_data["prompts"]
                print(f"Retrieved {len(previous_prompts)} previous prompts from memory")
        except Exception as e:
            print(f"Error retrieving previous prompts: {str(e)}")
    
    # Generate prompts
    try:
        prompt_generation = f"""
        Create {NUM_IMAGES} detailed Stable Diffusion XL prompts based on:
        
        DESCRIPTION: {description}
        
        {"PREVIOUS PROMPTS: " + json.dumps(previous_prompts) if previous_prompts else ""}
        
        Guidelines:
        - Be creative but precise
        - Include style, mood, lighting, details
        - Make each prompt unique but related to the theme
        - Format as SDXL understands (subject, style, details)
        
        Return ONLY a JSON array of strings, with each string being a complete prompt.
        """
        
        response = gemini_model.generate_content(prompt_generation)
        prompts = extract_json(response.text)
        
        # Ensure we have enough prompts
        while len(prompts) < NUM_IMAGES:
            prompts.append(f"{description}, detailed, high quality")
        
        # Limit to NUM_IMAGES
        prompts = prompts[:NUM_IMAGES]
        
        return {"image_prompts": prompts}
    except Exception as e:
        print(f"Error generating prompts: {str(e)}")
        # Fallback to simple prompts
        return {
            "image_prompts": [f"{description}, detailed, high quality"] * NUM_IMAGES
        }

def generate_images(state: ImageAgentState):
    """Generate images using Hugging Face's API"""
    prompts = state["image_prompts"]
    images = []
    image_paths = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n🖼️ Generating image {i+1}/{len(prompts)}...")
        
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
            
            response = requests.post(
                API_URL,
                headers=headers,
                json={"inputs": prompt}
            )
            
            if response.status_code == 200:
                # Generate unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}_{i+1}.png"
                file_path = os.path.join(OUTPUT_DIR, filename)
                
                img = Image.open(BytesIO(response.content))
                img.save(file_path)
                image_paths.append(file_path)
                
                images.append({
                    "prompt": prompt,
                    "path": file_path
                })
            else:
                print(f"Error: {response.status_code} - {response.text}")
                images.append({
                    "prompt": prompt,
                    "error": f"API Error: {response.status_code}"
                })
        
        except Exception as e:
            print(f"Failed to generate image: {e}")
            images.append({
                "prompt": prompt,
                "error": str(e)
            })
    
    return {"generated_images": images}

def store_in_memory(state: ImageAgentState):
    """Store the generated images in memory system"""
    try:
        description = state["dataset_description"]
        prompts = state["image_prompts"]
        images = state["generated_images"]
        
        print("\n📦 Storing image generation to memory system...")
        
        # Extract paths for successful generations
        paths = [img.get("path") for img in images if "path" in img]
        
        if memory_system and paths:
            memory_id = memory_system.store_image_data(
                description=description,
                prompts=prompts,
                image_paths=paths
            )
            print(f"✅ Images stored with ID: {memory_id}")
            return {"memory_id": memory_id}
        else:
            if not memory_system:
                print("⚠️ Memory system not available")
            if not paths:
                print("⚠️ No successful image generations to store")
            return {"memory_id": None}
    except Exception as e:
        print(f"❌ Error in store_to_memory: {str(e)}")
        return {"memory_id": None}

def format_output(state: ImageAgentState):
    """Prepare final output with summary"""
    description = state["dataset_description"]
    images = state["generated_images"]
    memory_id = state.get("memory_id")
    context_history = state.get("context_history", [])
    
    # Save metadata
    metadata = {
        "description": description,
        "prompts": state["image_prompts"],
        "images": images,
        "memory_id": memory_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Count successful generations
    success_count = sum(1 for img in images if "error" not in img)
    
    # Create output text
    output = f"""
    ====== IMAGE GENERATION RESULTS ======
    
    📋 Description: {description}
    
    ✅ Successfully generated: {success_count}/{len(images)} images
    📁 Output directory: {OUTPUT_DIR}/
    
    Example prompts used:
    1. {state['image_prompts'][0][:70]}{"..." if len(state['image_prompts'][0]) > 70 else ""}
    """
    
    if len(state['image_prompts']) > 1:
        output += f"\n    2. {state['image_prompts'][1][:70]}{'...' if len(state['image_prompts'][1]) > 70 else ''}"
    
    if memory_id:
        output += f"\n\n    🔖 Images stored with memory ID: {memory_id}"
        output += "\n    💡 You can reference these images in future requests."
    
    # Create a concise summary for context history
    summary = f"Generated {success_count} images of {description[:30]}{'...' if len(description) > 30 else ''}"
    
    return {
        "output": output,
        "summary": summary,
        "memory_id": memory_id  # Pass memory_id to be stored in context history
    }

def extract_json(text):
    """Robust JSON extraction from Gemini responses"""
    try:
        return json.loads(text)
    except:
        # Try to find JSON in code blocks
        if "```json" in text:
            try:
                json_text = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_text)
            except:
                pass
                
        # Try to find any JSON array or object
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            try:
                start = text.index(start_char)
                end = text.rindex(end_char) + 1
                return json.loads(text[start:end])
            except:
                continue
                
        # Return empty list as fallback
        print(f"Warning: Failed to parse JSON from response: {text[:100]}...")
        return []

# Define the workflow
workflow = StateGraph(ImageAgentState)

# Add nodes
workflow.add_node("analyze", analyze_request)
workflow.add_node("create_prompts", create_prompts)
workflow.add_node("generate_images", generate_images)
workflow.add_node("store_memory", store_in_memory)
workflow.add_node("format_output", format_output)

# Set edges
workflow.add_edge("analyze", "create_prompts")
workflow.add_edge("create_prompts", "generate_images")
workflow.add_edge("generate_images", "store_memory")
workflow.add_edge("store_memory", "format_output")
workflow.add_edge("format_output", END)

# Set the entrypoint
workflow.set_entry_point("analyze")

# Compile the graph into a runnable
agent = workflow.compile()

if __name__ == "__main__":
    print("=== Image Generation Agent ===")
    print("This agent creates images based on text descriptions")
    print("Type 'exit' to quit")
    
    # Initialize context history
    context_history = []
    
    while True:
        try:
            query = input("\nDescribe the images you want to generate: ")
            
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting image generation agent. Goodbye!")
                break
                
            if not query.strip():
                print("Please enter a valid query.")
                continue
                
            print("\n🚀 Processing your request...")
            
            # Pass the existing context history to the agent
            result = agent.invoke({
                "user_query": query,
                "context_history": context_history
            })
            
            # Update context history with this interaction
            new_interaction = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "agent_type": "image",
                "summary": result.get("summary", "Generated images"),
                "memory_id": result.get("memory_id")
            }
            
            # Add new interaction to context history
            context_history.append(new_interaction)
            if len(context_history) > 10:  # Keep only the last 10 interactions
                context_history = context_history[-10:]
            
            print("\n" + result["output"])
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again with a different request.")
