from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv
from data_memory import GeneratedDataMemory

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize memory system
try:
    memory_system = GeneratedDataMemory()
except Exception as e:
    print(f"Warning: Memory system initialization failed: {str(e)}")
    memory_system = None

try:
    model = genai.GenerativeModel('gemini-2.0-flash')
    test_response = model.generate_content("Hello")
    if not test_response.text:
        raise ConnectionError("Gemini API connection failed - no response text")
except Exception as e:
    raise ConnectionError(f"Failed to initialize Gemini API: {str(e)}")

class AgentState(TypedDict):
    user_query: str
    dataset_description: Annotated[str, "Description of the dataset to be generated"]
    dataset_schema: Annotated[dict, "Schema for the dataset"]
    generated_dataset: Annotated[List[dict], "The generated dataset entries"]
    memory_id: Annotated[str, "ID of the stored memory entry"]
    output: Annotated[str, "The final output to return to the user"]

def extract_json_from_response(response_text):
    """Helper function to extract JSON from Gemini response"""
    try:
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Response text was: {response_text}")
        return None

def parse_user_query(state: AgentState):
    """Parse the user query to understand what dataset they want"""
    try:
        user_query = state["user_query"]
        print(f"\n[DEBUG] Parsing user query: {user_query}")
        
        prompt = f"""
        Analyze this dataset generation request and return ONLY valid JSON:
        {{
          "data_type": "description of data",
          "fields": [{{"name": "field1", "type": "string"}}],
          "constraints": ["constraint1"]
        }}

        User Query: {user_query}
        """
        
        response = model.generate_content(prompt)
        print(f"[DEBUG] Raw API response: {response.text}")
        
        analysis = extract_json_from_response(response.text)
        if not analysis:
            raise ValueError("Failed to parse API response as JSON")
        
        print(f"[DEBUG] Parsed analysis: {analysis}")
        
        return {
            "dataset_description": analysis.get("data_type", user_query),
            "dataset_schema": {
                "fields": analysis.get("fields", [{"name": "field1", "type": "string"}]),
                "constraints": analysis.get("constraints", [])
            }
        }
    except Exception as e:
        print(f"[ERROR] in parse_user_query: {str(e)}")
        return {
            "dataset_description": user_query,
            "dataset_schema": {
                "fields": [{"name": "field1", "type": "string"}],
                "constraints": []
            }
        }

def generate_dataset_schema(state: AgentState):
    """Generate a detailed schema based on the initial analysis"""
    try:
        description = state["dataset_description"]
        initial_schema = state["dataset_schema"]
        print(f"\n[DEBUG] Generating schema from: {description}")
        
        prompt = f"""
        Create a complete dataset schema from this description. Return ONLY valid JSON:
        {{
          "description": "detailed description",
          "fields": [
            {{
              "name": "field_name",
              "type": "data_type",
              "description": "field purpose",
              "example": "example value"
            }}
          ],
          "constraints": ["constraint1"]
        }}

        Description: {description}
        Initial Schema: {json.dumps(initial_schema, indent=2)}
        """
        
        response = model.generate_content(prompt)
        print(f"[DEBUG] Raw schema response: {response.text}")
        
        enhanced_schema = extract_json_from_response(response.text)
        if not enhanced_schema:
            raise ValueError("Failed to parse schema response")
        
        print(f"[DEBUG] Generated schema: {enhanced_schema}")
        return {"dataset_schema": enhanced_schema}
    except Exception as e:
        print(f"[ERROR] in generate_dataset_schema: {str(e)}")
        return {"dataset_schema": initial_schema}

def generate_dataset_entries(state: AgentState):
    """Generate actual dataset entries based on the schema"""
    try:
        schema = state["dataset_schema"]
        num_entries = 5  # Reduced for testing
        print(f"\n[DEBUG] Generating {num_entries} entries from schema")
        
        prompt = f"""
        Generate {num_entries} dataset entries from this schema. Return ONLY valid JSON array:
        [
          {{
            "field1": "value1",
            "field2": "value2"
          }}
        ]

        Schema: {json.dumps(schema, indent=2)}
        """
        
        response = model.generate_content(prompt)
        print(f"[DEBUG] Raw data response: {response.text}")
        
        dataset = extract_json_from_response(response.text)
        if not dataset or not isinstance(dataset, list):
            raise ValueError("Invalid dataset format")
        
        print(f"[DEBUG] First entry: {dataset[0] if dataset else 'empty'}")
        return {"generated_dataset": dataset}
    except Exception as e:
        print(f"[ERROR] in generate_dataset_entries: {str(e)}")
        return {"generated_dataset": []}

def validate_dataset(state: AgentState):
    """Validate the generated dataset against the schema"""
    try:
        schema = state["dataset_schema"]
        dataset = state["generated_dataset"]
        if not dataset:
            return {"generated_dataset": []}
        
        print(f"\n[DEBUG] Validating {len(dataset)} entries")
        
        prompt = f"""
        Validate this dataset against its schema. Return ONLY valid JSON:
        {{
          "is_valid": true|false,
          "issues": ["issue1"],
          "validated_dataset": [...]  // Only if corrections needed
        }}

        Schema: {json.dumps(schema, indent=2)}
        Dataset: {json.dumps(dataset[:3], indent=2)}...  // First 3 entries shown
        """
        
        response = model.generate_content(prompt)
        print(f"[DEBUG] Raw validation response: {response.text}")
        
        validation = extract_json_from_response(response.text)
        if not validation:
            return {"generated_dataset": dataset}
        
        if not validation.get("is_valid", False):
            print(f"[WARNING] Validation issues: {validation.get('issues', [])}")
            return {"generated_dataset": validation.get("validated_dataset", dataset)}
        
        return {"generated_dataset": dataset}
    except Exception as e:
        print(f"[ERROR] in validate_dataset: {str(e)}")
        return {"generated_dataset": dataset}

def store_to_memory(state: AgentState):
    """Store the generated dataset in memory system"""
    try:
        description = state["dataset_description"]
        schema = state["dataset_schema"]
        dataset = state["generated_dataset"]
        
        print("\n[DEBUG] Storing dataset to memory system")
        
        if memory_system and dataset:
            memory_id = memory_system.store_text_data(
                description=description,
                generated_data=dataset,
                schema=schema
            )
            print(f"[DEBUG] Dataset stored with ID: {memory_id}")
            return {"memory_id": memory_id}
        else:
            if not memory_system:
                print("[WARNING] Memory system not available")
            if not dataset:
                print("[WARNING] No dataset to store")
            return {"memory_id": None}
    except Exception as e:
        print(f"[ERROR] in store_to_memory: {str(e)}")
        return {"memory_id": None}

def format_output(state: AgentState):
    """Format the final output for the user"""
    try:
        schema = state["dataset_schema"]
        dataset = state["generated_dataset"]
        memory_id = state.get("memory_id")
        
        output = f"""
        # Generated Dataset
        
        ## Schema Description
        {schema.get('description', 'No description available')}
        
        ## Fields
        """
        
        for field in schema.get("fields", []):
            output += f"\n- **{field.get('name', '')}**: {field.get('description', '')} ({field.get('type', '')})"
            if "example" in field:
                output += f" - Example: `{field['example']}`"
        
        output += "\n\n## Dataset Preview\n```json\n"
        output += json.dumps(dataset[:3], indent=2) if dataset else "[]"
        output += "\n```"
        
        output += f"\n\nTotal entries generated: {len(dataset) if dataset else 0}"
        
        if memory_id:
            output += f"\n\nDataset stored in memory with ID: {memory_id}"
            output += "\nYou can refer to this dataset in future queries."
        
        # Save to file
        try:
            with open("generated_dataset.json", "w") as f:
                json.dump({
                    "schema": schema,
                    "data": dataset if dataset else []
                }, f, indent=2)
            print("[DEBUG] Dataset saved to file")
        except Exception as e:
            print(f"[ERROR] saving dataset: {str(e)}")
        
        return {"output": output}
    except Exception as e:
        print(f"[ERROR] in format_output: {str(e)}")
        return {"output": "Error generating output"}

# Create and configure the workflow
workflow = StateGraph(AgentState)
workflow.add_node("parse_query", parse_user_query)
workflow.add_node("generate_schema", generate_dataset_schema)
workflow.add_node("generate_entries", generate_dataset_entries)
workflow.add_node("validate_data", validate_dataset)
workflow.add_node("store_memory", store_to_memory)
workflow.add_node("format_output", format_output)

workflow.add_edge("parse_query", "generate_schema")
workflow.add_edge("generate_schema", "generate_entries")
workflow.add_edge("generate_entries", "validate_data")
workflow.add_edge("validate_data", "store_memory")
workflow.add_edge("store_memory", "format_output")
workflow.add_edge("format_output", END)
workflow.set_entry_point("parse_query")

agent = workflow.compile()

if __name__ == "__main__":
    print("Starting dataset generation agent...")
    user_query = "I need a dataset of fictional books with title, author, genre, and year"
    
    try:
        print(f"\nProcessing query: '{user_query}'")
        result = agent.invoke({"user_query": user_query})
        
        if not result or "output" not in result:
            raise ValueError("Agent didn't return expected output format")
        
        print("\n=== FINAL OUTPUT ===")
        print(result["output"])
        print("\nDataset saved to 'generated_dataset.json'")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Agent failed: {str(e)}")
        print("Possible causes:")
        print("- Invalid Gemini API key (even if you think it's valid)")
        print("- Network connectivity issues")
        print("- Google API service outage")
        print("- Bug in the code (unlikely but possible)")
        print("\nCheck the debug output above for more clues.")
