import os
import random
import json
import time
from typing import Dict, List
import csv
from transformers import pipeline
import torch
from googlesearch import search
import chromadb
from sentence_transformers import SentenceTransformer

# Temporary storage for generated datasets
TEMP_STORAGE = "generated_datasets"
os.makedirs(TEMP_STORAGE, exist_ok=True)

class DatasetAgent:
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize the dataset agent with a local model.
        
        Args:
            model_name: Name of the local model to use (default: GPT-2)
        """
        # Load the local model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if device == "cuda" else -1
        )
        
        # Initialize ChromaDB for RAG
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="context_collection")
        
        # Load sentence transformer model for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Cache for generated values
        self.value_cache = {}

    def _retrieve_context(self, query: str) -> str:
        """
        Retrieve context using RAG (Retrieval-Augmented Generation) with Google Search and ChromaDB.
        """
        try:
            # Perform Google Search
            search_results = list(search(query, num=5, stop=5, pause=2))
            
            # Retrieve and store embeddings in ChromaDB
            for result in search_results:
                embedding = self.embedding_model.encode(result)
                self.collection.add(
                    documents=[result],
                    embeddings=[embedding.tolist()],
                    ids=[str(hash(result))]
                )
            
            # Query ChromaDB for relevant context
            query_embedding = self.embedding_model.encode(query)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=1
            )
            
            if results['documents']:
                return results['documents'][0][0]
            else:
                return "This should be a realistic data value appropriate for a database column."
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return "This should be a realistic data value appropriate for a database column."

    def _generate_value(self, column_name: str, context: str) -> str:
        """
        Generate a value using a local model with retrieved context.
        """
        prompt = f"""
        Generate a realistic value for the column '{column_name}' in a dataset.
        The column represents: {context}.
        Examples:
        - For 'product_name': 'iPhone 13 Pro', 'Samsung Galaxy S21', 'MacBook Air M2'.
        - For 'description': 'A high-performance smartphone with a 6.1-inch display and 5G support.',
        - For 'price': '999.99', '1499.99', '199.99'.
        - For 'category': 'Electronics', 'Clothing', 'Home'.
        """
        
        try:
            # Generate text using the local model
            response = self.generator(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                truncation=True,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )
            generated_text = response[0]["generated_text"].strip()
            
            # Extract the generated value (remove the prompt)
            generated_value = generated_text.replace(prompt, "").strip()
            return self._post_process_value(column_name, generated_value)
        except Exception as e:
            print(f"Generation error for {column_name}: {e}")
            return f"Sample {column_name} {random.randint(1, 1000)}"

    def _post_process_value(self, column_name: str, value: str) -> str:
        """
        Post-process the generated value to ensure it meets specific criteria.
        """
        if column_name == "price":
            try:
                price = float(value)
                return f"{price:.2f}"
            except ValueError:
                return f"{random.uniform(10.0, 500.0):.2f}"
        elif column_name == "product_name":
            if len(value.split()) > 5:  # Too long
                return "Sample Product"
            return value
        return value

    def parse_prompt(self, prompt: str) -> List[Dict]:
        """
        Parse the input prompt to extract dataset requirements.
        Example prompt: "Generate a dataset with 100 rows containing product names, descriptions, prices, and categories."
        """
        # Example parsing logic (can be enhanced with NLP)
        columns = []
        if "product names" in prompt.lower():
            columns.append({"name": "product_name", "type": "text"})
        if "descriptions" in prompt.lower():
            columns.append({"name": "description", "type": "text"})
        if "prices" in prompt.lower():
            columns.append({"name": "price", "type": "number", "min": 9.99, "max": 299.99})
        if "categories" in prompt.lower():
            columns.append({"name": "category", "type": "category", "options": ["Electronics", "Clothing", "Home", "Beauty", "Sports"]})
        
        return columns

    def generate_dataset(self, prompt: str, num_rows: int = 100, format: str = 'csv') -> str:
        """
        Generate a dataset based on the input prompt.
        
        Args:
            prompt: Input prompt describing the dataset requirements
            num_rows: Number of rows to generate
            format: Output format ('csv', 'json', or 'parquet')
            
        Returns:
            Path to the generated dataset file
        """
        # Parse the prompt to extract schema
        schema = self.parse_prompt(prompt)
        if not schema:
            raise ValueError("Could not parse dataset requirements from the prompt.")

        # Prepare file paths
        dataset_id = f"dataset_{int(time.time())}"
        dataset_dir = os.path.join(TEMP_STORAGE, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)
        file_path = os.path.join(dataset_dir, f"data.{format}")

        # Initialize file based on format
        if format == 'csv':
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[col['name'] for col in schema])
                writer.writeheader()
        elif format == 'json':
            with open(file_path, 'w') as f:
                f.write('[\n')  # Start JSON array

        # Generate data
        for row_idx in range(num_rows):
            row = {}
            for col in schema:
                col_name = col['name']
                col_type = col['type']
                context = self._retrieve_context(col_name)
                row[col_name] = self._generate_value(col_name, context)

            # Write row to file
            if format == 'csv':
                with open(file_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    writer.writerow(row)
            elif format == 'json':
                with open(file_path, 'a') as f:
                    if row_idx > 0:
                        f.write(',\n')
                    json.dump(row, f, indent=2)

        # Finalize JSON file if needed
        if format == 'json':
            with open(file_path, 'a') as f:
                f.write('\n]')

        # Validate the generated dataset
        self.validate_dataset(file_path)
        return file_path

    def validate_dataset(self, dataset_path: str):
        """
        Validate the generated dataset.
        """
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            
            # Check for missing values
            if df.isnull().any().any():
                print("Warning: Missing values found in the dataset.")
            
            # Check for unrealistic prices
            if "price" in df.columns:
                if (df["price"] <= 0).any():
                    print("Warning: Invalid prices found in the dataset.")
            
            print("Dataset validation complete.")
        except ImportError:
            print("Install pandas to validate the dataset.")

if __name__ == "__main__":
    # Initialize agent with a local model
    agent = DatasetAgent(model_name="gpt2")  # You can replace "gpt2" with any Hugging Face model
    
    # Example prompt
    prompt = "Generate a dataset with 20 rows containing product names, descriptions, prices, and categories."
    
    # Generate dataset
    dataset_path = agent.generate_dataset(prompt, num_rows=20, format='csv')
    
    print(f"Dataset generated at: {dataset_path}")
    
    # Display sample
    try:
        import pandas as pd
        print(f"Sample data:\n{pd.read_csv(dataset_path).head()}")
    except ImportError:
        print("Install pandas to view the dataset sample.")
        with open(dataset_path, 'r') as f:
            for i, line in enumerate(f):
                if i <= 5:  # Header + 5 rows
                    print(line.strip())
                else:
                    break
