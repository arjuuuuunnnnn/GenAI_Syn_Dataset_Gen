from datetime import datetime
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional, Union
import uuid

class GeneratedDataMemory:
    """System for storing and retrieving generated data with semantic search capabilities"""
    
    def __init__(self, persist_dir="./memory_db"):
        os.makedirs(persist_dir, exist_ok=True)
        os.makedirs("./output_history", exist_ok=True)
        
        print("Initializing data memory system...")
        try:
            # self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.db = chromadb.PersistentClient(path=persist_dir)
            embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.text_collection = self.db.get_or_create_collection(
                name="text_memory",
                embedding_function=embedding_function
            )
            self.image_collection = self.db.get_or_create_collection(
                name="image_memory",
                embedding_function=embedding_function
                )
            
            print("Memory system initialized successfully.")
        except Exception as e:
            print(f"Warning: Memory system initialization failed: {str(e)}")
            print("The system will function but without memory capabilities.")
            self.embed_model = None
            self.db = None
    
    def store_text_data(self, 
                       description: str, 
                       generated_data: List[Dict], 
                       schema: Dict = None) -> str:
        """Store generated text data with its description and schema"""
        if not self.db:
            return None
            
        try:
            memory_id = f"text_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now().isoformat()
            
            # Create summary for embedding
            summary = f"Text data: {description}. "
            if schema and 'fields' in schema:
                fields = [f["name"] for f in schema.get("fields", [])]
                summary += f"Fields: {', '.join(fields)}. "
            
            # Add sample data to summary
            if generated_data and len(generated_data) > 0:
                sample = generated_data[0]
                sample_str = ", ".join([f"{k}: {v}" for k, v in sample.items()])
                summary += f"Sample: {sample_str}"
            
            # Store full data to disk
            data_path = f"./output_history/{memory_id}.json"
            with open(data_path, "w") as f:
                json.dump({
                    "type": "text",
                    "description": description,
                    "schema": schema,
                    "data": generated_data,
                    "timestamp": timestamp
                }, f, indent=2)
            
            # Store in vector DB for search
            self.text_collection.add(
                documents=[summary],
                embeddings=[self.embed_model.encode(summary).tolist()],
                ids=[memory_id],
                metadatas=[{
                    "type": "text",
                    "description": description,
                    "timestamp": timestamp,
                    "path": data_path
                }]
            )
            
            return memory_id
            
        except Exception as e:
            print(f"Error storing text data: {str(e)}")
            return None
    
    def store_image_data(self, 
                        description: str, 
                        prompts: List[str], 
                        image_paths: List[str]) -> str:
        """Store generated image data with its description and paths"""
        if not self.db:
            return None
            
        try:
            memory_id = f"img_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now().isoformat()
            
            # Create summary for embedding
            summary = f"Image data: {description}. "
            if prompts and len(prompts) > 0:
                summary += f"Examples: {prompts[0]}"
            
            # Store full data to disk
            data_path = f"./output_history/{memory_id}.json"
            with open(data_path, "w") as f:
                json.dump({
                    "type": "image",
                    "description": description,
                    "prompts": prompts,
                    "image_paths": image_paths,
                    "timestamp": timestamp
                }, f, indent=2)
            
            # Store in vector DB for search
            self.image_collection.add(
                documents=[summary],
                embeddings=[self.embed_model.encode(summary).tolist()],
                ids=[memory_id],
                metadatas=[{
                    "type": "image",
                    "description": description,
                    "timestamp": timestamp,
                    "path": data_path
                }]
            )
            
            return memory_id
            
        except Exception as e:
            print(f"Error storing image data: {str(e)}")
            return None
    
    def search_memory(self, query: str, limit: int = 3) -> List[Dict]:
        """Search all memory for relevant data based on query"""
        if not self.db:
            return []
            
        try:
            query_embedding = self.embed_model.encode(query).tolist()
            
            # Search text collection
            text_results = self.text_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Search image collection
            image_results = self.image_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            # Combine results
            results = []
            
            if text_results and len(text_results['ids'][0]) > 0:
                for i, doc_id in enumerate(text_results['ids'][0]):
                    metadata = text_results['metadatas'][0][i]
                    results.append({
                        "id": doc_id,
                        "type": "text",
                        "description": metadata.get("description", ""),
                        "path": metadata.get("path", ""),
                        "summary": text_results['documents'][0][i],
                        "timestamp": metadata.get("timestamp", "")
                    })
            
            if image_results and len(image_results['ids'][0]) > 0:
                for i, doc_id in enumerate(image_results['ids'][0]):
                    metadata = image_results['metadatas'][0][i]
                    results.append({
                        "id": doc_id,
                        "type": "image",
                        "description": metadata.get("description", ""),
                        "path": metadata.get("path", ""),
                        "summary": image_results['documents'][0][i],
                        "timestamp": metadata.get("timestamp", "")
                    })
            
            # Sort by relevance (already done by the query)
            return results
            
        except Exception as e:
            print(f"Error searching memory: {str(e)}")
            return []
    
    def get_data_by_id(self, memory_id: str) -> Optional[Dict]:
        """Retrieve full data for a specific memory ID"""
        if not memory_id:
            return None
            
        try:
            # Find path from collections
            metadata = None
            if memory_id.startswith("text_"):
                result = self.text_collection.get(ids=[memory_id])
                if result and result['metadatas'] and len(result['metadatas']) > 0:
                    metadata = result['metadatas'][0]
            elif memory_id.startswith("img_"):
                result = self.image_collection.get(ids=[memory_id])
                if result and result['metadatas'] and len(result['metadatas']) > 0:
                    metadata = result['metadatas'][0]
            
            if not metadata or 'path' not in metadata:
                # Fall back to direct search
                data_path = f"./output_history/{memory_id}.json"
            else:
                data_path = metadata['path']
            
            # Load from disk
            if os.path.exists(data_path):
                with open(data_path, "r") as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            print(f"Error retrieving data by ID: {str(e)}")
            return None
    
    def get_recent_data(self, limit: int = 5) -> List[Dict]:
        """Get most recently generated data"""
        recent_files = []
        
        try:
            if os.path.exists("./output_history"):
                files = os.listdir("./output_history")
                json_files = [f for f in files if f.endswith('.json')]
                
                # Sort by modification time (most recent first)
                json_files.sort(key=lambda x: os.path.getmtime(os.path.join("./output_history", x)), reverse=True)
                
                # Get metadata for recent files
                for filename in json_files[:limit]:
                    file_path = os.path.join("./output_history", filename)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        
                    memory_id = filename.replace(".json", "")
                    recent_files.append({
                        "id": memory_id,
                        "type": data.get("type", "unknown"),
                        "description": data.get("description", "No description"),
                        "timestamp": data.get("timestamp", "Unknown time")
                    })
            
            return recent_files
            
        except Exception as e:
            print(f"Error getting recent data: {str(e)}")
            return []
    
    def get_human_readable_memory_summary(self, memory_data: Dict) -> str:
        """Create a human-readable summary of memory data"""
        if not memory_data:
            return "Data not found or could not be loaded."
        
        memory_type = memory_data.get("type", "unknown")
        description = memory_data.get("description", "No description")
        timestamp = memory_data.get("timestamp", "Unknown time")
        
        if memory_type == "text":
            schema = memory_data.get("schema", {})
            data = memory_data.get("data", [])
            
            summary = f"Text Dataset: {description}\n"
            summary += f"Generated on: {timestamp}\n"
            
            if schema and 'fields' in schema:
                summary += "\nFields:\n"
                for field in schema.get("fields", []):
                    field_name = field.get("name", "Unnamed")
                    field_type = field.get("type", "Unknown type")
                    field_desc = field.get("description", "No description")
                    summary += f"- {field_name} ({field_type}): {field_desc}\n"
            
            summary += f"\nTotal entries: {len(data)}\n"
            
            if data and len(data) > 0:
                summary += "\nSample data (first 2 entries):\n"
                for i, entry in enumerate(data[:2]):
                    summary += f"\nEntry {i+1}:\n"
                    for key, value in entry.items():
                        summary += f"  {key}: {value}\n"
            
            return summary
            
        elif memory_type == "image":
            prompts = memory_data.get("prompts", [])
            image_paths = memory_data.get("image_paths", [])
            
            summary = f"Image Generation: {description}\n"
            summary += f"Generated on: {timestamp}\n"
            summary += f"Total images: {len(image_paths)}\n"
            
            if prompts and len(prompts) > 0:
                summary += "\nPrompts used:\n"
                for i, prompt in enumerate(prompts[:3]):
                    summary += f"{i+1}. {prompt}\n"
                if len(prompts) > 3:
                    summary += f"... and {len(prompts) - 3} more prompts\n"
            
            summary += f"\nImage files stored at: {', '.join(image_paths[:3])}"
            if len(image_paths) > 3:
                summary += f" and {len(image_paths) - 3} more"
            
            return summary
        
        return f"Unknown data type: {memory_type}\nDescription: {description}"
