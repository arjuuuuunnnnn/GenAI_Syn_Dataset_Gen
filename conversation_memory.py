# conversation_memory.py
from typing import Dict, List, Any, Optional
import os
import json
import uuid
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

class ConversationMemory:
    """System for automatically storing and retrieving conversation history"""
    
    def __init__(self, persist_dir="./conversation_db"):
        os.makedirs(persist_dir, exist_ok=True)
        os.makedirs("./conversation_history", exist_ok=True)
        
        print("Initializing conversation memory system...")
        try:
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.db = chromadb.PersistentClient(path=persist_dir)
            self.conversation_collection = self.db.get_or_create_collection("conversation_memory")
            
            print("Conversation memory system initialized successfully.")
        except Exception as e:
            print(f"Warning: Conversation memory system initialization failed: {str(e)}")
            print("The system will function but without conversation memory capabilities.")
            self.embed_model = None
            self.db = None
    
    def store_interaction(self, user_query: str, system_response: str, agent_type: str) -> str:
        """Store a user-system interaction in memory"""
        if not self.db:
            return None
            
        try:
            memory_id = f"conv_{uuid.uuid4().hex[:8]}"
            timestamp = datetime.now().isoformat()
            
            # Create summary for embedding - this is what will be used for retrieval
            summary = f"Q: {user_query} A: {system_response[:200]}..."
            
            # Retrieve the last N interactions
            last_interactions = self.get_last_n_interactions(3)
            
            # Store full interaction to disk
            data_path = f"./conversation_history/{memory_id}.json"
            with open(data_path, "w") as f:
                json.dump({
                    "type": "conversation",
                    "agent_type": agent_type,
                    "query": user_query,
                    "response": system_response,
                    "timestamp": timestamp,
                    "last_interactions": last_interactions
                }, f, indent=2)
            
            # Store in vector DB for search
            self.conversation_collection.add(
                documents=[summary],
                embeddings=[self.embed_model.encode(summary).tolist()],
                ids=[memory_id],
                metadatas=[{
                    "type": "conversation",
                    "agent_type": agent_type,
                    "timestamp": timestamp,
                    "path": data_path,
                    "query": user_query[:100]  # Store a preview of the query in metadata
                }]
            )
            
            return memory_id
            
        except Exception as e:
            print(f"Error storing conversation: {str(e)}")
            return None
    
    def search_conversation_history(self, query: str, limit: int = 3) -> List[Dict]:
        """Search conversation history for relevant interactions"""
        if not self.db:
            return []
            
        try:
            query_embedding = self.embed_model.encode(query).tolist()
            
            # Search conversation collection
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            conversations = []
            
            if results and len(results['ids'][0]) > 0:
                for i, conv_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Load full conversation data from disk
                    path = metadata.get("path", "")
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            conversation_data = json.load(f)
                    else:
                        conversation_data = {
                            "query": metadata.get("query", ""),
                            "response": "Full response not available"
                        }
                    
                    conversations.append({
                        "id": conv_id,
                        "query": conversation_data.get("query", ""),
                        "response": conversation_data.get("response", ""),
                        "agent_type": metadata.get("agent_type", "unknown"),
                        "timestamp": metadata.get("timestamp", "")
                    })
            
            return conversations
            
        except Exception as e:
            print(f"Error searching conversation history: {str(e)}")
            return []

    def get_last_n_interactions(self, n: int) -> List[Dict]:
        """Retrieve the last N interactions from the conversation history"""
        if not self.db:
            return []
        
        try:
            # Retrieve the last N interactions based on timestamp
            results = self.conversation_collection.get(
                sort_by="timestamp",
                sort_order="desc",
                limit=n
            )
            
            interactions = []
            
            if results and len(results['ids']) > 0:
                for i, conv_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    
                    # Load full conversation data from disk
                    path = metadata.get("path", "")
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            conversation_data = json.load(f)
                    else:
                        conversation_data = {
                            "query": metadata.get("query", ""),
                            "response": "Full response not available"
                        }
                    
                    interactions.append({
                        "id": conv_id,
                        "query": conversation_data.get("query", ""),
                        "response": conversation_data.get("response", ""),
                        "agent_type": metadata.get("agent_type", "unknown"),
                        "timestamp": metadata.get("timestamp", "")
                    })
            
            return interactions
            
        except Exception as e:
            print(f"Error retrieving last {n} interactions: {str(e)}")
            return []
