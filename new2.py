import argparse
import csv
import os
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from tqdm import tqdm
import re
import random
import datetime
from pydantic import BaseModel, Field, validator
import faker

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
                 use_rag: bool = False, knowledge_dir: Optional[str] = None,
                 use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            # Initialize with HuggingFaceHub - free model
            try:
                self.llm = HuggingFaceHub(
                    repo_id=model_name,
                    model_kwargs={"temperature": temperature, "max_length": 2048}
                )
                print(f"Successfully initialized {model_name}")
            except Exception as e:
                print(f"Warning: Failed to initialize LLM: {e}")
                print("Falling back to direct generation methods")
                self.use_llm = False
        
        # Initialize Faker for direct generation
        self.fake = faker.Faker()
        
        # Set up categorical values dictionary
        self.categorical_values = {
            "gender": ["Male", "Female", "Non-binary", "Prefer not to say"],
            "payment_type": ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash"],
            "status": ["Active", "Pending", "Completed", "Cancelled", "Failed"],
            "country": ["USA", "Canada", "UK", "Germany", "France", "Japan", "Australia", "India", "Brazil", "Mexico"],
            "category": ["Electronics", "Clothing", "Food", "Books", "Sports", "Home", "Beauty", "Toys", "Automotive", "Health"],
            "priority": ["Low", "Medium", "High", "Critical"],
            "user_type": ["Free", "Premium", "Enterprise", "Admin"],
            "subscription": ["Monthly", "Quarterly", "Annual", "Lifetime"]
        }
        
        self.use_rag = use_rag
        self.retriever = None
        
        if use_rag and knowledge_dir and use_llm:
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
        if self.use_llm:
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
                return self._extract_config_from_text(user_prompt)
        else:
            # Direct parsing without LLM
            return self._extract_config_from_text(user_prompt)
    
    def _extract_config_from_text(self, text: str) -> DatasetConfig:
        """Extract configuration from text using heuristics"""
        # Try to guess column types from keywords in the text
        columns = []
        
        # Look for common column patterns
        column_patterns = [
            (r'\b(?:name|username|full[_ ]name|first[_ ]name|last[_ ]name)\b', "name"),
            (r'\b(?:email|e-mail|email[_ ]address)\b', "email"),
            (r'\b(?:phone|telephone|mobile|cell|phone[_ ]number)\b', "phone"),
            (r'\b(?:address|location|street|city|state|country|zip|postal)\b', "address"),
            (r'\b(?:date|time|day|month|year|birthday|birth[_ ]date|created[_ ]at|updated[_ ]at)\b', "date"),
            (r'\b(?:id|uuid|identifier|key)\b', "id"),
            (r'\b(?:price|cost|amount|value|salary|income|expense|revenue|profit|loss|money)\b', "number"),
            (r'\b(?:quantity|count|number|age|height|weight|score|rating|rank)\b', "number"),
            (r'\b(?:description|comment|message|content|text|note|summary|detail)\b', "text"),
            (r'\b(?:type|category|status|gender|level|priority|state|class)\b', "categorical"),
            (r'\b(?:is[_ ]|has[_ ]|active|enabled|completed|verified|approved|flagged)\b', "boolean")
        ]
        
        # Check for explicit column mentions
        col_matches = re.findall(r'\b(\w+)[_ ](?:column|field)\b', text)
        for col in col_matches:
            # Try to guess the type
            col_type = "text"  # Default
            for pattern, type_name in column_patterns:
                if re.search(pattern, col.lower()):
                    col_type = type_name
                    break
            
            columns.append({
                "name": col,
                "type": col_type,
                "description": f"Auto-detected {col} column"
            })
        
        # Look for words that might be column names
        if not columns:
            potential_columns = re.findall(r'\b(\w+)\b', text)
            for col in potential_columns:
                # Skip common words
                if col.lower() in ["with", "and", "the", "for", "dataset", "create", "generate", "rows", "columns"]:
                    continue
                    
                # Try to guess the type
                col_type = "text"  # Default
                for pattern, type_name in column_patterns:
                    if re.search(pattern, col.lower()):
                        col_type = type_name
                        break
                
                columns.append({
                    "name": col,
                    "type": col_type,
                    "description": f"Auto-detected {col} column"
                })
                
                # Limit to 5 auto-detected columns
                if len(columns) >= 5:
                    break
        
        # If no columns found, use default
        if not columns:
            columns = [
                {"name": "name", "type": "name", "description": "Person's name"},
                {"name": "email", "type": "email", "description": "Email address"},
                {"name": "age", "type": "number", "description": "Age in years"},
                {"name": "is_active", "type": "boolean", "description": "Active status"}
            ]
        
        # Look for number of rows
        num_rows = 10  # Default
        row_match = re.search(r'(\d+)[_ ](?:rows|records|entries)', text)
        if row_match:
            num_rows = int(row_match.group(1))
        
        return DatasetConfig(columns=columns, num_rows=num_rows)
    
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
    
    def _generate_value_directly(self, column_name: str, column_type: str) -> Any:
        """Generate a single value for a column directly using Faker"""
        column_name_lower = column_name.lower()
        
        # Handle by type
        if column_type == "name":
            return self.fake.name()
        elif column_type == "email":
            return self.fake.email()
        elif column_type == "phone":
            return self.fake.phone_number()
        elif column_type == "address":
            if "street" in column_name_lower:
                return self.fake.street_address()
            elif "city" in column_name_lower:
                return self.fake.city()
            elif "state" in column_name_lower:
                return self.fake.state()
            elif "country" in column_name_lower:
                return self.fake.country()
            elif "zip" in column_name_lower or "postal" in column_name_lower:
                return self.fake.zipcode()
            else:
                return self.fake.address()
        elif column_type == "date":
            if "birth" in column_name_lower:
                return self.fake.date_of_birth().strftime("%Y-%m-%d")
            elif "created" in column_name_lower:
                return self.fake.date_time_this_year().strftime("%Y-%m-%d %H:%M:%S")
            else:
                return self.fake.date_this_decade().strftime("%Y-%m-%d")
        elif column_type == "number":
            if "age" in column_name_lower:
                return random.randint(18, 85)
            elif "price" in column_name_lower or "cost" in column_name_lower or "amount" in column_name_lower:
                return round(random.uniform(10, 1000), 2)
            elif "id" in column_name_lower:
                return random.randint(1000, 9999)
            else:
                return random.randint(1, 100)
        elif column_type == "boolean":
            return random.choice([True, False])
        elif column_type == "categorical":
            for category_name, values in self.categorical_values.items():
                if category_name in column_name_lower:
                    return random.choice(values)
            return random.choice(["Option A", "Option B", "Option C"])
        elif column_type == "id":
            if "uuid" in column_name_lower:
                return str(self.fake.uuid4())
            else:
                return f"{random.randint(1000, 9999)}-{random.randint(100, 999)}"
        elif column_type == "text":
            if "description" in column_name_lower or "comment" in column_name_lower:
                return self.fake.paragraph(nb_sentences=3)
            else:
                return self.fake.sentence()
        else:
            return self.fake.word()
    
    def generate_rows(self, config: DatasetConfig) -> List[Dict[str, Any]]:
        """Generate rows of data based on the configuration"""
        column_names = [col["name"] for col in config.columns]
        column_types = {col["name"]: col["type"] for col in config.columns}
        column_descriptions = {col["name"]: col.get("description", "") for col in config.columns}
        
        # Generate all rows
        rows = []
        print(f"Generating {config.num_rows} rows of data...")
        
        for _ in tqdm(range(config.num_rows)):
            row_data = {}
            
            # Generate each column value
            for name in column_names:
                col_type = column_types[name]
                
                # Always use direct generation for efficiency and consistency
                value = self._generate_value_directly(name, col_type)
                row_data[name] = value
                
            rows.append(row_data)
        
        return rows
    
    def generate_dataset(self, user_prompt: str) -> pd.DataFrame:
        """Generate a complete dataset based on the user prompt"""
        # Parse the configuration from the prompt
        config = self.parse_prompt(user_prompt)
        
        # Print the configuration
        print("Detected configuration:")
        print(f"Number of rows: {config.num_rows}")
        print("Columns:")
        for col in config.columns:
            print(f"  - {col['name']} ({col['type']}): {col.get('description', '')}")
        
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
    parser.add_argument("--no-llm", action="store_true", help="Don't use LLM, use direct generation only")
    
    args = parser.parse_args()
    
    use_llm = not args.no_llm
    if use_llm and "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        print("NOTE: You need a Hugging Face API token for LLM use (it's free!)")
        print("Get it at: https://huggingface.co/settings/tokens")
        print("Alternatively, use --no-llm to use direct generation only")
        api_key = input("Please enter your Hugging Face API token (or press Enter to use direct generation): ")
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        else:
            use_llm = False
    
    # Create generator
    generator = DatasetGenerator(
        model_name=args.model,
        temperature=args.temperature,
        use_rag=args.rag,
        knowledge_dir=args.knowledge_dir if args.rag else None,
        use_llm=use_llm
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
