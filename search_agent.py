import os
import json
import time
from typing import Dict, List, Optional, Any, TypedDict
from langgraph.graph import END, StateGraph
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Define our state as a TypedDict for LangGraph
class AgentState(TypedDict):
    query: str
    dataset_type: Optional[str]
    domain: Optional[str]
    dataset_suggestions: Optional[List[Dict[str, Any]]]
    final_response: Optional[str]
    error: Optional[str]

def initialize_state(query: str) -> AgentState:
    return {
        "query": query,
        "dataset_type": None,
        "domain": None,
        "dataset_suggestions": None,
        "final_response": None,
        "error": None
    }

def extract_json_from_response(response_text: str) -> Dict[str, str]:
    """Extract JSON from various possible response formats."""
    try:
        # Try direct JSON parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Handle cases where response might be wrapped in markdown
        if '```json' in response_text:
            json_str = response_text.split('```json')[1].split('```')[0].strip()
            return json.loads(json_str)
        elif '```' in response_text:
            json_str = response_text.split('```')[1].split('```')[0].strip()
            return json.loads(json_str)
        else:
            # Try to find JSON within the text
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                return json.loads(response_text[start:end])
            except:
                raise ValueError("Could not extract JSON from response")

def analyze_query(state: AgentState) -> AgentState:
    try:
        prompt = """Analyze this dataset request and respond ONLY with JSON containing:
        {
            "dataset_type": "image/text/tabular/etc",
            "domain": "medical/finance/education/etc"
        }

        Request: {query}""".format(query=state['query'])

        # Add delay to avoid rate limiting
        time.sleep(1)
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Debug print the raw response
        print(f"Raw API response: {response_text}")
        
        result = extract_json_from_response(response_text)
        
        if not isinstance(result, dict):
            raise ValueError("Response is not a JSON object")
        
        return {
            **state,
            "dataset_type": result.get("dataset_type", "unknown"),
            "domain": result.get("domain", "unknown")
        }
    except Exception as e:
        error_msg = f"Error analyzing query: {str(e)}"
        print(f"Analysis error: {error_msg}")
        return {
            **state,
            "error": error_msg
        }


# Helper functions to search different dataset repositories
def find_datasets_kaggle(dataset_type: Optional[str], domain: Optional[str]) -> List[Dict[str, Any]]:
    try:
        prompt = f"""You are a Kaggle dataset expert. 
        The user is looking for {dataset_type} datasets in the {domain} domain.
        Return a list of 2-3 real, free, open-source datasets from Kaggle that match this criteria.
        
        For each dataset, provide:
        1. Name
        2. Brief description (1-2 sentences)
        3. URL (use the format: https://www.kaggle.com/datasets/[username]/[dataset-name])
        4. Size (approximate)
        5. Format (CSV, JSON, images, etc.)
        
        Respond in JSON format as a list of datasets."""
        
        response = model.generate_content(prompt)
        results = json.loads(response.text)
        for result in results:
            result["source"] = "Kaggle"
        return results
    except Exception as e:
        print(f"Error with Kaggle search: {str(e)}")
        return []

def find_datasets_huggingface(dataset_type: Optional[str], domain: Optional[str]) -> List[Dict[str, Any]]:
    try:
        prompt = f"""You are a Hugging Face dataset expert. 
        The user is looking for {dataset_type} datasets in the {domain} domain.
        Return a list of 2-3 real, free, open-source datasets from Hugging Face that match this criteria.
        
        For each dataset, provide:
        1. Name
        2. Brief description (1-2 sentences)
        3. URL (use the format: https://huggingface.co/datasets/[dataset-name])
        4. Size (approximate)
        5. Format
        
        Respond in JSON format as a list of datasets."""
        
        response = model.generate_content(prompt)
        results = json.loads(response.text)
        for result in results:
            result["source"] = "Hugging Face"
        return results
    except Exception as e:
        print(f"Error with Hugging Face search: {str(e)}")
        return []

def find_datasets_google_dataset_search(dataset_type: Optional[str], domain: Optional[str]) -> List[Dict[str, Any]]:
    try:
        prompt = f"""You are a Google Dataset Search expert. 
        The user is looking for {dataset_type} datasets in the {domain} domain.
        Return a list of 2-3 real, free, open-source datasets from Google Dataset Search that match this criteria.
        
        For each dataset, provide:
        1. Name
        2. Brief description (1-2 sentences)
        3. URL
        4. Size (approximate)
        5. Format
        
        Respond in JSON format as a list of datasets."""
        
        response = model.generate_content(prompt)
        results = json.loads(response.text)
        for result in results:
            result["source"] = "Google Dataset Search"
        return results
    except Exception as e:
        print(f"Error with Google Dataset Search: {str(e)}")
        return []

def find_datasets_openml(dataset_type: Optional[str], domain: Optional[str]) -> List[Dict[str, Any]]:
    if not dataset_type or dataset_type.lower() not in ["tabular", "csv", "machine learning"]:
        return []
    
    try:
        prompt = f"""You are an OpenML dataset expert. 
        The user is looking for {dataset_type} datasets in the {domain} domain.
        Return a list of 1-2 real, free, open-source datasets from OpenML that match this criteria.
        
        For each dataset, provide:
        1. Name
        2. Brief description (1-2 sentences)
        3. URL (use the format: https://www.openml.org/d/[dataset-id])
        4. Size (approximate)
        5. Format
        
        Respond in JSON format as a list of datasets."""
        
        response = model.generate_content(prompt)
        results = json.loads(response.text)
        for result in results:
            result["source"] = "OpenML"
        return results
    except Exception as e:
        print(f"Error with OpenML search: {str(e)}")
        return []

def find_datasets_uci(dataset_type: Optional[str], domain: Optional[str]) -> List[Dict[str, Any]]:
    try:
        prompt = f"""You are a UCI Machine Learning Repository expert. 
        The user is looking for {dataset_type} datasets in the {domain} domain.
        Return a list of 1-2 real, free, open-source datasets from UCI ML Repository that match this criteria.
        
        For each dataset, provide:
        1. Name
        2. Brief description (1-2 sentences)
        3. URL (use the format: https://archive.ics.uci.edu/dataset/[dataset-name])
        4. Size (approximate)
        5. Format
        
        Respond in JSON format as a list of datasets."""
        
        response = model.generate_content(prompt)
        results = json.loads(response.text)
        for result in results:
            result["source"] = "UCI ML Repository"
        return results
    except Exception as e:
        print(f"Error with UCI search: {str(e)}")
        return []

# Node 2: Find datasets from various sources
def find_datasets(state: AgentState) -> AgentState:
    if state.get('error'):
        return state
    
    sources = [
        find_datasets_kaggle,
        find_datasets_huggingface,
        find_datasets_google_dataset_search,
        find_datasets_openml,
        find_datasets_uci
    ]
    
    all_datasets = []
    for source_func in sources:
        try:
            datasets = source_func(state.get('dataset_type'), state.get('domain'))
            if datasets:
                all_datasets.extend(datasets)
        except Exception as e:
            print(f"Error with {source_func.__name__}: {str(e)}")
    
    return {
        **state,
        "dataset_suggestions": all_datasets[:7] if all_datasets else None
    }

def generate_response(state: AgentState) -> AgentState:
    if state.get('error'):
        return {
            **state,
            "final_response": "I couldn't process your request. Please try again with a different query."
        }
    
    if not state.get('dataset_suggestions'):
        return {
            **state,
            "final_response": "I couldn't find any matching datasets. Try broadening your search."
        }
    
    try:
        prompt = """Create a helpful response listing these datasets:
        {datasets}
        
        Query: {query}
        Domain: {domain}
        Type: {dtype}
        
        Format:
        1. Introduction
        2. Dataset list with names, descriptions, and sources
        3. Recommendation""".format(
            datasets=json.dumps(state['dataset_suggestions'], indent=2),
            query=state['query'],
            domain=state.get('domain', 'general'),
            dtype=state.get('dataset_type', 'various')
        )
        
        response = model.generate_content(prompt)
        return {
            **state,
            "final_response": response.text
        }
    except Exception as e:
        return {
            **state,
            "final_response": "Here are some datasets I found:\n" + 
            "\n".join(f"{d['name']} ({d['source']})" for d in state['dataset_suggestions'])
        }

def create_dataset_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("find_datasets", find_datasets)
    workflow.add_node("generate_response", generate_response)
    
    workflow.add_edge("analyze_query", "find_datasets")
    workflow.add_edge("find_datasets", "generate_response")
    workflow.add_edge("generate_response", END)
    
    workflow.set_entry_point("analyze_query")
    
    return workflow.compile()

def run_dataset_agent(query: str) -> str:
    if not query or not isinstance(query, str):
        return "Please provide a valid search query."
    
    print(f"\nProcessing query: '{query}'")
    try:
        agent = create_dataset_agent()
        result = agent.invoke(initialize_state(query))
        
        if result.get('error'):
            print(f"Error in processing: {result['error']}")
        
        return result.get("final_response", "Sorry, I couldn't process your request.")
    except Exception as e:
        print(f"System error: {str(e)}")
        return "Our system encountered an error. Please try again later."

if __name__ == "__main__":
    print("Dataset Search Agent (type 'exit' to quit)")
    while True:
        try:
            query = input("\nEnter your dataset search query: ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                print("Please enter a query.")
                continue
                
            response = run_dataset_agent(query)
            print("\nResults:")
            print(response)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
