import os
import random
import json
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

TEMP_STORAGE = "generated_datasets"
os.makedirs(TEMP_STORAGE, exist_ok=True)

class TextDatasetGenerator:
    """Generate text-based datasets using LangChain"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional API key"""
        self.api_key = api_key
        
        self.text_generation_prompt = PromptTemplate(
            input_variables=["column_name", "context"],
            template="""
            Generate a realistic and contextually appropriate value for a column named '{column_name}' in a dataset.
            Additional context: {context}
            
            The text should be concise, realistic, and appropriate for a dataset column.
            Generate only the text value without any explanations or quotation marks.
            """
        )
        
    def _create_llm(self, temperature: float = 0.7):
        """Create LLM with specified temperature"""
        return OpenAI(
            temperature=temperature,
            openai_api_key=self.api_key
        )
    
    def generate_text(self, column_name: str, context: str = "") -> str:
        """Generate text using LLM"""
        llm = self._create_llm()
        chain = LLMChain(llm=llm, prompt=self.text_generation_prompt)
        return chain.run(column_name=column_name, context=context)
    
    @staticmethod
    def generate_number(col_def: Dict) -> float:
        """Generate a random number based on column definition"""
        min_val = col_def.get('min', 0)
        max_val = col_def.get('max', 100)
        
        if all(isinstance(x, int) for x in [min_val, max_val]):
            return random.randint(min_val, max_val)
        else:
            return random.uniform(min_val, max_val)

    @staticmethod
    def generate_category(col_def: Dict) -> str:
        """Generate a random category from options"""
        return random.choice(col_def.get('options', ['Unknown']))
    
    def generate_dataset(self, schema: List[Dict], num_rows: int = 100, 
                         format: str = 'csv', with_data: bool = False) -> Union[str, Tuple[str, List[Dict]]]:
        """Generate a dataset based on schema and save to specified format"""
        data = []
        
        # Generate context for the dataset to make related columns consistent
        overall_context = f"Creating a dataset with {num_rows} rows and columns: {', '.join([col['name'] for col in schema])}"
        
        for i in range(num_rows):
            if i % 10 == 0:
                print(f"Generating row {i+1}/{num_rows}")
                
            row = {}
            
            for col in schema:
                if col['type'] == 'text':
                    # Generate text based on column name and overall context
                    row[col['name']] = self.generate_text(col['name'], overall_context).strip()
                elif col['type'] == 'number':
                    row[col['name']] = self.generate_number(col)
                elif col['type'] == 'category':
                    row[col['name']] = self.generate_category(col)
                else:
                    # Default fallback
                    row[col['name']] = f"Unknown type: {col['type']}"
            
            data.append(row)
            
        # Save the dataset
        file_path = self._save_dataset(data, schema, format)
        
        if with_data:
            return file_path, data
        else:
            return file_path
    
    def _save_dataset(self, data: List[Dict], schema: List[Dict], format: str) -> str:
        """Save the generated dataset to disk"""
        # Create a unique ID for this dataset
        import hashlib
        schema_str = json.dumps(schema, sort_keys=True)
        dataset_id = hashlib.md5(schema_str.encode()).hexdigest()[:10]
        
        dataset_dir = os.path.join(TEMP_STORAGE, f"text_dataset_{dataset_id}")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save dataset in specified format
        file_path = os.path.join(dataset_dir, f"data.{format}")
        
        if format == 'csv':
            pd.DataFrame(data).to_csv(file_path, index=False)
        elif format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'parquet':
            pd.DataFrame(data).to_parquet(file_path, index=False)
        else:
            # Default to CSV if format not supported
            file_path = os.path.join(dataset_dir, "data.csv")
            pd.DataFrame(data).to_csv(file_path, index=False)
        
        # Also save schema information
        schema_path = os.path.join(dataset_dir, "schema.json")
        with open(schema_path, 'w') as f:
            json.dump({
                "columns": schema,
                "num_rows": len(data),
                "format": format
            }, f, indent=2)
        
        return file_path

# Simple test if run directly
if __name__ == "__main__":
    generator = TextDatasetGenerator()
    
    test_schema = [
        {"name": "name", "type": "text"},
        {"name": "age", "type": "number", "min": 18, "max": 80},
        {"name": "city", "type": "category", "options": ["New York", "London", "Tokyo", "Paris"]}
    ]
    
    dataset_path = generator.generate_dataset(
        schema=test_schema,
        num_rows=5,
        format='csv'
    )
    
    print(f"Dataset generated at: {dataset_path}")
    print(f"Sample data:\n{pd.read_csv(dataset_path)}")
