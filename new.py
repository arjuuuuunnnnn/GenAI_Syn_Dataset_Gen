import argparse
import csv
import os
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from tqdm import tqdm
import re
from pydantic import BaseModel, Field, validator

# Import for Hugging Face models
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

class DatasetConfig(BaseModel):
    columns: List[Dict[str, str]] = Field(description="List of column configurations with name, type, and description")
    num_rows: int = Field(description="Number of rows to generate")
    
    @validator('columns')
    def validate_columns(cls, columns):
        valid_types = ["text", "number", "date", "boolean", "categorical", "email", "phone", "address", "name", "id"]
        for col in columns:
            if "name" not in col:
                raise ValueError(f"Column missing 'name' field: {col}")
            if "type" not in col:
                raise ValueError(f"Column missing 'type' field: {col}")
            if col["type"] not in valid_types:
                raise ValueError(f"Invalid column type: {col['type']}. Must be one of {valid_types}")
        return columns

class DatasetGenerator:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", temperature: float = 0.7, 
                 use_rag: bool = False, knowledge_dir: Optional[str] = None):
        # Initialize with HuggingFaceHub - free model
        self.llm = HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={"temperature": temperature, "max_length": 2048}
        )
        self.use_rag = use_rag
        self.retriever = None
        
        if use_rag and knowledge_dir:
            self._setup_rag(knowledge_dir)
    
    def _setup_rag(self, knowledge_dir: str):
        """Set up RAG with documents from the specified directory"""
        if not os.path.exists(knowledge_dir):
            print(f"Warning: Knowledge directory {knowledge_dir} does not exist. RAG will not be used.")
            return
            
        try:
            # Load documents
            loader = DirectoryLoader(knowledge_dir, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            
            # Split documents
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            # Create vector store with free Hugging Face embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(texts, embeddings)
            
            # Create retriever
            self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            print(f"RAG setup complete with {len(texts)} document chunks.")
        except Exception as e:
            print(f"Failed to set up RAG: {e}")
            self.use_rag = False
    
    def _parse_json_from_text(self, text: str) -> Dict:
        """Extract JSON from model output text"""
        # Find JSON-like structure in the text
        pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
        # If no valid JSON found, try extracting key-value pairs
        result = {}
        lines = text.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip().strip('"\'')
                value = parts[1].strip().strip('"\'')
                if key and value:
                    result[key] = value
                    
        return result if result else {"error": "Could not parse response"}
    
    def parse_prompt(self, user_prompt: str) -> DatasetConfig:
        """Parse user prompt to get dataset configuration"""
        template = """
        Based on the following request, create a structured dataset configuration in JSON format.
        Extract the column names, types, and any descriptions from the request.
        Valid column types are: text, number, date, boolean, categorical, email, phone, address, name, id.
        
        Request: {user_prompt}
        
        Output the JSON in the following format:
        {{
            "columns": [
                {{"name": "column_name1", "type": "column_type1", "description": "column_description1"}},
                {{"name": "column_name2", "type": "column_type2", "description": "column_description2"}}
            ],
            "num_rows": number_of_rows_to_generate
        }}
        
        If the number of rows is not specified in the request, default to 10 rows.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["user_prompt"]
        )
        
        # Get the formatted prompt
        formatted_prompt = prompt.format(user_prompt=user_prompt)
        
        # Get response from LLM
        response = self.llm.predict(formatted_prompt)
        
        try:
            # Parse the JSON from the response
            config_dict = self._parse_json_from_text(response)
            
            # Check if required fields exist
            if "columns" not in config_dict:
                config_dict["columns"] = []
            if "num_rows" not in config_dict:
                config_dict["num_rows"] = 10
                
            # Create and validate the config
            config = DatasetConfig(**config_dict)
            return config
        except Exception as e:
            print(f"Failed to parse configuration: {e}")
            print("Raw LLM response:", response)
            # Create default config if parsing fails
            return DatasetConfig(columns=[
                {"name": "column1", "type": "text", "description": "Default column"},
                {"name": "column2", "type": "number", "description": "Default column"}
            ], num_rows=5)
    
    def _get_context_from_rag(self, query: str) -> str:
        """Retrieve relevant context using RAG"""
        if not self.use_rag or not self.retriever:
            return ""
            
        try:
            docs = self.retriever.get_relevant_documents(query)
            if docs:
                context = "\n\n".join(doc.page_content for doc in docs)
                return f"\nRelevant context for generation:\n{context}\n"
            return ""
        except Exception as e:
            print(f"RAG retrieval failed: {e}")
            return ""
    
    def _generate_single_value(self, column_name: str, column_type: str, column_desc: str) -> Any:
        """Generate a single value for a column based on its type"""
        prompt = f"Generate a realistic {column_type} value for a column named '{column_name}'"
        if column_desc:
            prompt += f" described as '{column_desc}'"
            
        try:
            response = self.llm.predict(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating value for {column_name}: {e}")
            return f"Error: {column_type}"
    
    def generate_rows(self, config: DatasetConfig) -> List[Dict[str, Any]]:
        """Generate rows of data based on the configuration"""
        column_names = [col["name"] for col in config.columns]
        column_types = {col["name"]: col["type"] for col in config.columns}
        column_descriptions = {col["name"]: col.get("description", "") for col in config.columns}
        
        # Create a template for generating an entire row at once
        template = """
        Generate a single realistic data row for a dataset with the following columns:
        
        {column_details}
        
        {rag_context}
        
        Output the row as a JSON object mapping column names to values.
        Make sure each value is appropriate for its type.
        Create values that are realistic, consistent with each other, and appropriate for their column types.
        """
        
        column_details = "\n".join([
            f"- {name} ({column_types[name]}): {column_descriptions[name]}"
            for name in column_names
        ])
        
        # Get RAG context for the dataset in general
        rag_context = self._get_context_from_rag(f"Generate realistic data for {column_details}")
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["column_details", "rag_context"]
        )
        
        formatted_prompt = prompt.format(column_details=column_details, rag_context=rag_context)
        
        # Generate all rows
        rows = []
        print(f"Generating {config.num_rows} rows of data...")
        for _ in tqdm(range(config.num_rows)):
            try:
                # First attempt to generate the entire row as JSON
                response = self.llm.predict(formatted_prompt)
                row_data = self._parse_json_from_text(response)
                
                # Check if we got all columns, if not, fill in missing ones
                for name in column_names:
                    if name not in row_data:
                        row_data[name] = self._generate_single_value(
                            name, column_types[name], column_descriptions[name]
                        )
                        
                rows.append(row_data)
            except Exception as e:
                print(f"Failed to generate row: {e}")
                # Generate row column by column as fallback
                row_data = {}
                for name in column_names:
                    row_data[name] = self._generate_single_value(
                        name, column_types[name], column_descriptions[name]
                    )
                rows.append(row_data)
        
        return rows
    
    def generate_dataset(self, user_prompt: str) -> pd.DataFrame:
        """Generate a complete dataset based on the user prompt"""
        # Parse the configuration from the prompt
        config = self.parse_prompt(user_prompt)
        
        # Generate rows
        rows = self.generate_rows(config)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_file: str):
        """Save the generated dataset to a CSV file"""
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator (Using Free Models)")
    parser.add_argument("prompt", help="Description of the dataset to generate")
    parser.add_argument("-m", "--model", default="mistralai/Mistral-7B-Instruct-v0.2", 
                      help="HuggingFace model to use (default: mistralai/Mistral-7B-Instruct-v0.2)")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("-o", "--output", default="generated_data.csv", help="Output CSV file")
    parser.add_argument("--rag", action="store_true", help="Use RAG for more realistic data generation")
    parser.add_argument("--knowledge-dir", default="knowledge", help="Directory containing knowledge documents for RAG")
    
    args = parser.parse_args()
    
    # Ensure HUGGINGFACEHUB_API_TOKEN is set
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        print("NOTE: You need a Hugging Face API token (it's free!)")
        print("Get it at: https://huggingface.co/settings/tokens")
        api_key = input("Please enter your Hugging Face API token: ")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    # Create generator
    generator = DatasetGenerator(
        model_name=args.model,
        temperature=args.temperature,
        use_rag=args.rag,
        knowledge_dir=args.knowledge_dir if args.rag else None
    )
    
    # Generate dataset
    print(f"Parsing prompt: {args.prompt}")
    df = generator.generate_dataset(args.prompt)
    
    # Display sample
    print("\nSample of generated data:")
    print(df.head())
    
    # Save to CSV
    generator.save_to_csv(df, args.output)
    
    print(f"\nGenerated {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
