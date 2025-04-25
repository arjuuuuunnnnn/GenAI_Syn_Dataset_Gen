# from typing import TypedDict, Annotated, List, Dict, Any
# from langgraph.graph import StateGraph, END
# import google.generativeai as genai
# import os
# import json
# from dotenv import load_dotenv
# from data_memory import GeneratedDataMemory

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("Please set GEMINI_API_KEY in your .env file")

# genai.configure(api_key=GEMINI_API_KEY)

# # Initialize memory system
# try:
#     memory_system = GeneratedDataMemory()
# except Exception as e:
#     print(f"Warning: Memory system initialization failed: {str(e)}")
#     memory_system = None

# try:
#     model = genai.GenerativeModel('gemini-2.0-flash')
#     test_response = model.generate_content("Hello")
#     if not test_response.text:
#         raise ConnectionError("Gemini API connection failed - no response text")
# except Exception as e:
#     raise ConnectionError(f"Failed to initialize Gemini API: {str(e)}")

# class AgentState(TypedDict):
#     user_query: str
#     dataset_description: Annotated[str, "Description of the dataset to be generated"]
#     dataset_schema: Annotated[dict, "Schema for the dataset"]
#     generated_dataset: Annotated[List[dict], "The generated dataset entries"]
#     memory_id: Annotated[str, "ID of the stored memory entry"]
#     output: Annotated[str, "The final output to return to the user"]

# def extract_json_from_response(response_text):
#     """Helper function to extract JSON from Gemini response"""
#     try:
#         cleaned = response_text.strip()
#         if cleaned.startswith('```json'):
#             cleaned = cleaned[7:]
#         if cleaned.endswith('```'):
#             cleaned = cleaned[:-3]
#         cleaned = cleaned.strip()
        
#         return json.loads(cleaned)
#     except json.JSONDecodeError as e:
#         print(f"JSON parsing error: {str(e)}")
#         print(f"Response text was: {response_text}")
#         return None

# def parse_user_query(state: AgentState):
#     """Parse the user query to understand what dataset they want"""
#     try:
#         user_query = state["user_query"]
#         print(f"\n[DEBUG] Parsing user query: {user_query}")
        
#         prompt = f"""
#         Analyze this dataset generation request and return ONLY valid JSON:
#         {{
#           "data_type": "description of data",
#           "fields": [{{"name": "field1", "type": "string"}}],
#           "constraints": ["constraint1"]
#         }}

#         User Query: {user_query}
#         """
        
#         response = model.generate_content(prompt)
#         print(f"[DEBUG] Raw API response: {response.text}")
        
#         analysis = extract_json_from_response(response.text)
#         if not analysis:
#             raise ValueError("Failed to parse API response as JSON")
        
#         print(f"[DEBUG] Parsed analysis: {analysis}")
        
#         return {
#             "dataset_description": analysis.get("data_type", user_query),
#             "dataset_schema": {
#                 "fields": analysis.get("fields", [{"name": "field1", "type": "string"}]),
#                 "constraints": analysis.get("constraints", [])
#             }
#         }
#     except Exception as e:
#         print(f"[ERROR] in parse_user_query: {str(e)}")
#         return {
#             "dataset_description": user_query,
#             "dataset_schema": {
#                 "fields": [{"name": "field1", "type": "string"}],
#                 "constraints": []
#             }
#         }

# def generate_dataset_schema(state: AgentState):
#     """Generate a detailed schema based on the initial analysis"""
#     try:
#         description = state["dataset_description"]
#         initial_schema = state["dataset_schema"]
#         print(f"\n[DEBUG] Generating schema from: {description}")
        
#         prompt = f"""
#         Create a complete dataset schema from this description. Return ONLY valid JSON:
#         {{
#           "description": "detailed description",
#           "fields": [
#             {{
#               "name": "field_name",
#               "type": "data_type",
#               "description": "field purpose",
#               "example": "example value"
#             }}
#           ],
#           "constraints": ["constraint1"]
#         }}

#         Description: {description}
#         Initial Schema: {json.dumps(initial_schema, indent=2)}
#         """
        
#         response = model.generate_content(prompt)
#         print(f"[DEBUG] Raw schema response: {response.text}")
        
#         enhanced_schema = extract_json_from_response(response.text)
#         if not enhanced_schema:
#             raise ValueError("Failed to parse schema response")
        
#         print(f"[DEBUG] Generated schema: {enhanced_schema}")
#         return {"dataset_schema": enhanced_schema}
#     except Exception as e:
#         print(f"[ERROR] in generate_dataset_schema: {str(e)}")
#         return {"dataset_schema": initial_schema}

# def generate_dataset_entries(state: AgentState):
#     """Generate actual dataset entries based on the schema"""
#     try:
#         schema = state["dataset_schema"]
#         num_entries = 5  # Reduced for testing
#         print(f"\n[DEBUG] Generating {num_entries} entries from schema")
        
#         prompt = f"""
#         Generate {num_entries} dataset entries from this schema. Return ONLY valid JSON array:
#         [
#           {{
#             "field1": "value1",
#             "field2": "value2"
#           }}
#         ]

#         Schema: {json.dumps(schema, indent=2)}
#         """
        
#         response = model.generate_content(prompt)
#         print(f"[DEBUG] Raw data response: {response.text}")
        
#         dataset = extract_json_from_response(response.text)
#         if not dataset or not isinstance(dataset, list):
#             raise ValueError("Invalid dataset format")
        
#         print(f"[DEBUG] First entry: {dataset[0] if dataset else 'empty'}")
#         return {"generated_dataset": dataset}
#     except Exception as e:
#         print(f"[ERROR] in generate_dataset_entries: {str(e)}")
#         return {"generated_dataset": []}

# def validate_dataset(state: AgentState):
#     """Validate the generated dataset against the schema"""
#     try:
#         schema = state["dataset_schema"]
#         dataset = state["generated_dataset"]
#         if not dataset:
#             return {"generated_dataset": []}
        
#         print(f"\n[DEBUG] Validating {len(dataset)} entries")
        
#         prompt = f"""
#         Validate this dataset against its schema. Return ONLY valid JSON:
#         {{
#           "is_valid": true|false,
#           "issues": ["issue1"],
#           "validated_dataset": [...]  // Only if corrections needed
#         }}

#         Schema: {json.dumps(schema, indent=2)}
#         Dataset: {json.dumps(dataset[:3], indent=2)}...  // First 3 entries shown
#         """
        
#         response = model.generate_content(prompt)
#         print(f"[DEBUG] Raw validation response: {response.text}")
        
#         validation = extract_json_from_response(response.text)
#         if not validation:
#             return {"generated_dataset": dataset}
        
#         if not validation.get("is_valid", False):
#             print(f"[WARNING] Validation issues: {validation.get('issues', [])}")
#             return {"generated_dataset": validation.get("validated_dataset", dataset)}
        
#         return {"generated_dataset": dataset}
#     except Exception as e:
#         print(f"[ERROR] in validate_dataset: {str(e)}")
#         return {"generated_dataset": dataset}

# def store_to_memory(state: AgentState):
#     """Store the generated dataset in memory system"""
#     try:
#         description = state["dataset_description"]
#         schema = state["dataset_schema"]
#         dataset = state["generated_dataset"]
        
#         print("\n[DEBUG] Storing dataset to memory system")
        
#         if memory_system:
#             memory_id = memory_system.store_text_data(
#                 description=description,
#                 schema=schema,
#                 generated_data=dataset
#             )
#             print(f"[DEBUG] Dataset stored with ID: {memory_id}")
#             return {"memory_id": memory_id}
#         else:
#             print("[WARNING] Memory system not available")
#             return {"memory_id": None}
#     except Exception as e:
#         print(f"[ERROR] in store_to_memory: {str(e)}")
#         return {"memory_id": None}

# def format_output(state: AgentState):
#     """Format the final output for the user"""
#     try:
#         schema = state["dataset_schema"]
#         dataset = state["generated_dataset"]
#         memory_id = state.get("memory_id")
        
#         output = f"""
#         # Generated Dataset
        
#         ## Schema Description
#         {schema.get('description', 'No description available')}
        
#         ## Fields
#         """
        
#         for field in schema.get("fields", []):
#             output += f"\n- **{field.get('name', '')}**: {field.get('description', '')} ({field.get('type', '')})"
#             if "example" in field:
#                 output += f" - Example: `{field['example']}`"
        
#         output += "\n\n## Dataset Preview\n```json\n"
#         output += json.dumps(dataset[:3], indent=2) if dataset else "[]"
#         output += "\n```"
        
#         output += f"\n\nTotal entries generated: {len(dataset) if dataset else 0}"
        
#         if memory_id:
#             output += f"\n\nDataset stored in memory with ID: {memory_id}"
#             output += "\nYou can refer to this dataset in future queries."
        
#         # Save to file
#         try:
#             with open("generated_dataset.json", "w") as f:
#                 json.dump({
#                     "schema": schema,
#                     "data": dataset if dataset else []
#                 }, f, indent=2)
#             print("[DEBUG] Dataset saved to file")
#         except Exception as e:
#             print(f"[ERROR] saving dataset: {str(e)}")
        
#         return {"output": output}
#     except Exception as e:
#         print(f"[ERROR] in format_output: {str(e)}")
#         return {"output": "Error generating output"}

# # Create and configure the workflow
# workflow = StateGraph(AgentState)
# workflow.add_node("parse_query", parse_user_query)
# workflow.add_node("generate_schema", generate_dataset_schema)
# workflow.add_node("generate_entries", generate_dataset_entries)
# workflow.add_node("validate_data", validate_dataset)
# workflow.add_node("store_memory", store_to_memory)
# workflow.add_node("format_output", format_output)

# workflow.add_edge("parse_query", "generate_schema")
# workflow.add_edge("generate_schema", "generate_entries")
# workflow.add_edge("generate_entries", "validate_data")
# workflow.add_edge("validate_data", "store_memory")
# workflow.add_edge("store_memory", "format_output")
# workflow.add_edge("format_output", END)
# workflow.set_entry_point("parse_query")

# agent = workflow.compile()

# if __name__ == "__main__":
#     print("Starting dataset generation agent...")
#     user_query = "I need a dataset of fictional books with title, author, genre, and year"
    
#     try:
#         print(f"\nProcessing query: '{user_query}'")
#         result = agent.invoke({"user_query": user_query})
        
#         if not result or "output" not in result:
#             raise ValueError("Agent didn't return expected output format")
        
#         print("\n=== FINAL OUTPUT ===")
#         print(result["output"])
#         print("\nDataset saved to 'generated_dataset.json'")
#     except Exception as e:
#         print(f"\n[CRITICAL ERROR] Agent failed: {str(e)}")
#         print("Possible causes:")
#         print("- Invalid Gemini API key (even if you think it's valid)")
#         print("- Network connectivity issues")
#         print("- Google API service outage")
#         print("- Bug in the code (unlikely but possible)")
#         print("\nCheck the debug output above for more clues.")




from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
import google.generativeai as genai
import os
import json
import csv
import re
from dotenv import load_dotenv
from data_memory import GeneratedDataMemory
import datetime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize text model
try:
    text_model = genai.GenerativeModel('gemini-2.0-flash')
    test_response = text_model.generate_content("Hello")
    if not test_response.text:
        raise ConnectionError("Gemini API connection failed - no response text")
except Exception as e:
    raise ConnectionError(f"Failed to initialize Gemini API: {str(e)}")

# Initialize memory system
try:
    memory_system = GeneratedDataMemory()
except Exception as e:
    print(f"Warning: Memory system initialization failed: {str(e)}")
    memory_system = None

# Create output directory if it doesn't exist
OUTPUT_DIR = "generated_content"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class TextAgentState(TypedDict):
    user_query: str
    context_history: Annotated[List[Dict[str, Any]], "History of previous interactions"]
    output: Annotated[str, "The final output to return to the user"]
    summary: Annotated[str, "Brief summary of the generated content"]
    file_path: Annotated[str, "Path to the saved output file"]
    format: Annotated[str, "Format of the saved data (json, csv, txt)"]

def sanitize_filename(text):
    """Create a safe filename from text"""
    # Replace non-alphanumeric chars with underscore
    safe = re.sub(r'[^\w\s-]', '_', text)
    # Replace whitespace with underscore
    safe = re.sub(r'[\s]+', '_', safe)
    # Trim to reasonable length
    return safe[:50]

def detect_format(text, user_query):
    """Detect the appropriate format based on content and query"""
    # Check if query explicitly requests a format
    if re.search(r'\b(json|JSON)\b', user_query):
        return "json"
    elif re.search(r'\b(csv|CSV)\b', user_query):
        return "csv"
    
    # Analyze content to detect format
    if re.search(r'^\s*\[.*\]\s*$', text, re.DOTALL) or re.search(r'^\s*\{.*\}\s*$', text, re.DOTALL):
        # Looks like JSON array or object
        return "json"
    elif re.search(r'.*,.*,.*\n', text):
        # Multiple commas in a line could indicate CSV
        return "csv"
    
    # Default to JSON for structured data, text for prose
    if re.search(r'\b(list|data|dataset|records|items)\b', user_query.lower()):
        return "json"
    else:
        return "txt"

def extract_json_from_text(text):
    """Extract JSON from text with potential markdown code blocks"""
    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        # Try to find JSON without code blocks
        json_text = text.strip()
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Try to extract any JSON-like structure
        try:
            matches = re.findall(r'(\[.*\]|\{.*\})', json_text, re.DOTALL)
            if matches:
                for potential_json in matches:
                    try:
                        return json.loads(potential_json)
                    except:
                        continue
        except:
            pass
        return None

def extract_csv_from_text(text):
    """Extract CSV data from text"""
    lines = text.strip().split('\n')
    csv_data = []
    
    # Try to detect and strip markdown table format
    if any('|' in line for line in lines[:5]):
        # Extract markdown table
        table_lines = [line for line in lines if '|' in line]
        # Remove the separator line if present (contains only |, -, and spaces)
        table_lines = [line for line in table_lines if not re.match(r'^[\s\|\-]+$', line)]
        # Convert to CSV format
        csv_lines = [','.join([cell.strip() for cell in line.split('|') if cell.strip()]) for line in table_lines]
        lines = csv_lines
    
    # Process as CSV
    for line in lines:
        if ',' in line:
            csv_data.append(line.split(','))
    
    if len(csv_data) > 0:
        return csv_data
    return None

def save_as_json(data, file_path):
    """Save data as JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        if isinstance(data, str):
            # Try to parse as JSON first
            try:
                parsed_data = json.loads(data)
                json.dump(parsed_data, f, indent=2)
            except json.JSONDecodeError:
                # Fall back to storing as string in JSON
                json.dump({"content": data}, f, indent=2)
        else:
            # Already a Python data structure
            json.dump(data, f, indent=2)
    return file_path

def save_as_csv(data, file_path):
    """Save data as CSV file"""
    try:
        if isinstance(data, list) and isinstance(data[0], list):
            # Already in CSV format (list of lists)
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(data)
        elif isinstance(data, list) and isinstance(data[0], dict):
            # List of dictionaries - common JSON data structure
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                if data:
                    fieldnames = data[0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data)
        else:
            # Fallback - convert to string and save
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
    except Exception as e:
        print(f"[ERROR] Failed to save as CSV: {str(e)}")
        # Fallback to text
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(str(data))
    return file_path

def generate_text_content(state: TextAgentState):
    """Generate text content based on user query and context history"""
    try:
        user_query = state["user_query"]
        context_history = state.get("context_history", [])
        
        print(f"\n[DEBUG] Generating text for query: {user_query}")
        
        # Filter only text-related previous interactions
        text_context = [
            interaction for interaction in context_history 
            if interaction.get("agent_type") == "text"
        ]
        
        # Build context information from relevant history
        context_info = ""
        if text_context:
            context_info = "Previously generated text content:\n"
            for i, interaction in enumerate(text_context[-3:]):  # Only use most recent 3 text interactions
                query = interaction.get("query", "")
                summary = interaction.get("summary", "")
                output = interaction.get("output", "")
                
                # Include abbreviated output (first few lines)
                output_lines = output.split("\n")[:5]
                short_output = "\n".join(output_lines)
                if len(output_lines) < len(output.split("\n")):
                    short_output += "\n..."
                    
                context_info += f"Query {i+1}: '{query}'\n"
                context_info += f"Summary: {summary}\n"
                context_info += f"Sample output: {short_output}\n\n"
        
        # Generate text content with context awareness
        prompt = f"""
        {context_info}
        
        Generate text content based on the following query:
        {user_query}
        
        If the query refers to previous data or datasets, incorporate that information in your response.
        If generating structured data, format it appropriately (CSV, JSON, etc.).
        If specifically creating data that needs structure, please output in valid JSON format.
        
        Also provide a brief summary (one sentence) of what you generated to use for future reference.
        Format your response as follows:
        
        SUMMARY: [brief one-line summary of what you generated]
        
        [main content generation - dataset, text, etc.]
        """
        
        response = text_model.generate_content(prompt)
        response_text = response.text
        print(f"[DEBUG] Got response from text model, length: {len(response_text)}")
        
        # Extract summary if available
        summary = "Generated text content"
        if "SUMMARY:" in response_text:
            summary_line = re.search(r'SUMMARY:.*?(?:\n|$)', response_text)
            if summary_line:
                summary = summary_line.group(0).replace("SUMMARY:", "").strip()
                # Remove the summary line from the output
                response_text = response_text.replace(summary_line.group(0), "", 1).strip()
        
        print(f"[DEBUG] Summary: {summary}")
        
        # Detect the appropriate format for saving
        output_format = detect_format(response_text, user_query)
        print(f"[DEBUG] Detected format: {output_format}")
        
        # Create timestamp and base filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"text_{timestamp}_{sanitize_filename(summary)}"
        
        # Process and save content based on format
        file_path = ""
        metadata = {
            "query": user_query,
            "summary": summary,
            "timestamp": datetime.datetime.now().isoformat(),
            "format": output_format
        }
        
        try:
            if output_format == "json":
                # Try to extract JSON data
                json_data = extract_json_from_text(response_text)
                if json_data:
                    # Save the extracted JSON
                    file_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
                    save_as_json({
                        "metadata": metadata,
                        "data": json_data
                    }, file_path)
                else:
                    # Fall back to saving the raw text in a JSON wrapper
                    file_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
                    save_as_json({
                        "metadata": metadata,
                        "content": response_text
                    }, file_path)
            
            elif output_format == "csv":
                # Try to extract CSV data
                csv_data = extract_csv_from_text(response_text)
                if csv_data:
                    # Save as CSV
                    file_path = os.path.join(OUTPUT_DIR, f"{base_filename}.csv")
                    save_as_csv(csv_data, file_path)
                    
                    # Also save metadata in a companion JSON file
                    metadata_path = os.path.join(OUTPUT_DIR, f"{base_filename}_metadata.json")
                    save_as_json(metadata, metadata_path)
                else:
                    # Fall back to JSON
                    file_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
                    save_as_json({
                        "metadata": metadata,
                        "content": response_text
                    }, file_path)
            
            else:  # txt format
                # Save as plain text with metadata header
                file_path = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"# Generated Text Content\n")
                    f.write(f"# Query: {user_query}\n")
                    f.write(f"# Summary: {summary}\n")
                    f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(response_text)
                
                # Also save metadata in a companion JSON file
                metadata_path = os.path.join(OUTPUT_DIR, f"{base_filename}_metadata.json")
                save_as_json(metadata, metadata_path)
            
            print(f"[DEBUG] Content saved to file: {file_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save content to file: {str(e)}")
            # Fallback save as text
            file_path = os.path.join(OUTPUT_DIR, f"{base_filename}_fallback.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response_text)
        
        # Store to memory if available
        memory_id = None
        if memory_system:
            try:
                memory_id = memory_system.store_text_data(
                    description=summary,
                    content_type="text",
                    generated_data=response_text
                )
                print(f"[DEBUG] Content stored in memory with ID: {memory_id}")
            except Exception as e:
                print(f"[WARNING] Failed to store in memory: {str(e)}")
        
        # Add memory reference and file path to the output
        output_with_info = response_text.strip()
        
        if memory_id:
            output_with_info += f"\n\nContent stored in memory with ID: {memory_id}"
            
        output_with_info += f"\n\nSaved to file: {file_path} (Format: {output_format})"
            
        return {
            "output": output_with_info,
            "summary": summary,
            "file_path": file_path,
            "format": output_format
        }
        
    except Exception as e:
        print(f"[ERROR] in generate_text_content: {str(e)}")
        return {
            "output": f"Error generating text content: {str(e)}",
            "summary": "Error in text generation",
            "file_path": "",
            "format": "error"
        }

# Define the workflow for the text generation agent
workflow = StateGraph(TextAgentState)

# Add nodes
workflow.add_node("generate_text", generate_text_content)

# Set edges
workflow.add_edge("generate_text", END)

# Set the entrypoint
workflow.set_entry_point("generate_text")

# Compile the graph into a runnable
agent = workflow.compile()

if __name__ == "__main__":
    print("Starting text generation agent...")
    user_query = "Generate a list of 10 fictional cities with population, country, and main industry in JSON format"
    
    try:
        print(f"\nProcessing query: '{user_query}'")
        result = agent.invoke({
            "user_query": user_query,
            "context_history": []
        })
        
        if not result or "output" not in result:
            raise ValueError("Agent didn't return expected output format")
        
        print("\n=== FINAL OUTPUT ===")
        print(result["output"])
        print("\nSummary:", result.get("summary", "No summary available"))
        if result.get("file_path"):
            print(f"\nContent saved to: {result['file_path']} (Format: {result.get('format', 'unknown')})")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Text agent failed: {str(e)}")
        print("Check the debug output above for more clues.")