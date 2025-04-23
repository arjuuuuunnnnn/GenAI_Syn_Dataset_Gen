from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import requests
import os
import json
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from data_memory import GeneratedDataMemory

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# Initialize memory system
try:
    memory_system = GeneratedDataMemory()
except Exception as e:
    print(f"Warning: Memory system initialization failed: {str(e)}")
    memory_system = None

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
NUM_IMAGES = 4
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AgentState(TypedDict):
    user_query: str
    dataset_description: str
    image_prompts: List[str]
    generated_images: List[dict]
    memory_id: Annotated[str, "ID of the stored memory entry"]
    output: str

def analyze_with_gemini(state: AgentState):
    """Use Gemini to understand the user's request"""
    user_query = state["user_query"]
    print("\n🔍 Analyzing your request with Gemini...")
    
    response = gemini_model.generate_content(
        f"Analyze this image request and return JSON with description, styles, and requirements:\n{user_query}\n"
        "Example: {'description':'...','styles':[...],'requirements':[...]}"
    )
    
    analysis = extract_json(response.text)
    return {
        "dataset_description": analysis.get("description", user_query),
        "image_prompts": []
    }

def create_prompts_with_gemini(state: AgentState):
    """Generate detailed SDXL prompts using Gemini"""
    description = state["dataset_description"]
    print(f"\n💡 Generating {NUM_IMAGES} prompts with Gemini...")
    
    response = gemini_model.generate_content(
        f"Create {NUM_IMAGES} Stable Diffusion prompts based on:\n{description}\n"
        "Return ONLY a JSON array of strings."
    )
    
    return {"image_prompts": extract_json(response.text)}

def generate_with_huggingface(state: AgentState):
    """Generate images using Hugging Face's free API"""
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
                img = Image.open(BytesIO(response.content))
                filename = f"image_{i+1}.png"
                file_path = f"{OUTPUT_DIR}/{filename}" 
                img.save(file_path)
                image_paths.append(file_path)
                
                images.append({
                    "prompt": prompt,
                    "path": file_path
                })
            else:
                print(f"Error: {response.text}")
                images.append({
                    "prompt": prompt,
                    "error": response.text
                })
        
        except Exception as e:
            print(f"Failed to generate image: {e}")
            images.append({
                "prompt": prompt,
                "error": str(e)
            })
    
    return {"generated_images": images}

def store_to_memory(state: AgentState):
    """Store the generated images in memory system"""
    try:
        description = state["dataset_description"]
        prompts = state["image_prompts"]
        images = state["generated_images"]
        
        print("\n[DEBUG] Storing image generation to memory system")
        
        # Extract paths for successful generations
        paths = [img.get("path") for img in images if "path" in img]
        
        if memory_system and paths:
            memory_id = memory_system.store_image_data(
                description=description,
                prompts=prompts,
                image_paths=paths
            )
            print(f"[DEBUG] Images stored with ID: {memory_id}")
            return {"memory_id": memory_id}
        else:
            if not memory_system:
                print("[WARNING] Memory system not available")
            if not paths:
                print("[WARNING] No successful image generations to store")
            return {"memory_id": None}
    except Exception as e:
        print(f"[ERROR] in store_to_memory: {str(e)}")
        return {"memory_id": None}

def format_output(state: AgentState):
    """Prepare final report"""
    metadata = {
        "description": state["dataset_description"],
        "images": state["generated_images"]
    }
    
    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)
    
    success_count = sum(1 for img in state["generated_images"] if "error" not in img)
    memory_id = state.get("memory_id")
    
    output = f"""
    🎉 Completed!
    - Successful generations: {success_count}/{len(state['generated_images'])}
    - Output directory: {OUTPUT_DIR}/
    - First prompt: {state['image_prompts'][0][:70]}...
    """
    
    if memory_id:
        output += f"\n    - Images stored in memory with ID: {memory_id}"
        output += "\n    - You can refer to these images in future queries."
    
    return {"output": output}

def extract_json(text):
    """Robust JSON extraction from Gemini responses"""
    try:
        return json.loads(text)
    except:
        for start_char, end_char in [('[', ']'), ('{', '}')]:
            try:
                start = text.index(start_char)
                end = text.rindex(end_char) + 1
                return json.loads(text[start:end])
            except:
                continue
    return []

workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_with_gemini)
workflow.add_node("generate_prompts", create_prompts_with_gemini)
workflow.add_node("generate_images", generate_with_huggingface)
workflow.add_node("store_memory", store_to_memory)
workflow.add_node("format", format_output)

workflow.add_edge("analyze", "generate_prompts")
workflow.add_edge("generate_prompts", "generate_images")
workflow.add_edge("generate_images", "store_memory")
workflow.add_edge("store_memory", "format")
workflow.add_edge("format", END)

workflow.set_entry_point("analyze")
agent = workflow.compile()

if __name__ == "__main__":
    query = input("Describe the images you want to generate: ")
    result = agent.invoke({"user_query": query})
    print(result["output"])
