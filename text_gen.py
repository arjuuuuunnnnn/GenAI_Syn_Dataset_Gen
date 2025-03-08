import os
import re
import csv
import json
import random
from typing import Dict, List
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import pandas as pd

TEMP_STORAGE = "generated_datasets"
os.makedirs(TEMP_STORAGE, exist_ok=True)

class DatasetSchema:
    def __init__(self, query: str):
        self.columns = self._parse_query(query)
        
    def _parse_query(self, query: str) -> List[Dict]:
        columns = []
        patterns = [
            r'"(\w+)"\s*\((\w+)(?:\s*range:\s*([\d\-]+))?(?:\s*options:\s*([\w\s,]+))?)?\)',
            r'column\s+(\w+)\s+as\s+(\w+)(?:\s+with\s+(.*))?'
        ]
        
        for match in re.finditer(patterns[0], query):
            name, dtype = match.group(1), match.group(2).lower()
            col_def = {'name': name, 'type': dtype}
            
            if dtype == 'number':
                if match.group(3):
                    min_val, max_val = map(int, match.group(3).split('-'))
                    col_def.update({'min': min_val, 'max': max_val})
            elif dtype == 'category' and match.group(4):
                options = [x.strip() for x in match.group(4).split(',')]
                col_def['options'] = options
                
            columns.append(col_def)
            
        return columns

class DataGenerator:
    @staticmethod
    def generate_number(col_def: Dict) -> float:
        if 'min' in col_def and 'max' in col_def:
            return random.uniform(col_def['min'], col_def['max'])
        return random.random()

    @staticmethod
    def generate_category(col_def: Dict) -> str:
        return random.choice(col_def.get('options', ['Unknown']))

class TextDatasetAgent:
    def __init__(self):
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        self.retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact")
        self.model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

    def generate_text(self, context: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(context, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class DatasetOrchestrator:
    def __init__(self):
        self.text_agent = TextDatasetAgent()
        self.data_gen = DataGenerator()
        
    def generate_dataset(self, query: str, num_rows: int = 100, format: str = 'csv') -> str:
        schema = DatasetSchema(query)
        data = []
        
        for _ in range(num_rows):
            row = {}
            for col in schema.columns:
                if col['type'] == 'text':
                    row[col['name']] = self.text_agent.generate_text(col['name'])
                elif col['type'] == 'number':
                    row[col['name']] = self.data_gen.generate_number(col)
                elif col['type'] == 'category':
                    row[col['name']] = self.data_gen.generate_category(col)
            data.append(row)
            
        return self._save_dataset(data, schema, format)

    def _save_dataset(self, data: List[Dict], schema: DatasetSchema, format: str) -> str:
        dataset_id = f"dataset_{hash(str(schema.columns))}"
        os.makedirs(os.path.join(TEMP_STORAGE, dataset_id), exist_ok=True)
        
        filename = os.path.join(TEMP_STORAGE, dataset_id, f"data.{format}")
        
        if format == 'csv':
            pd.DataFrame(data).to_csv(filename, index=False)
        elif format == 'json':
            pd.DataFrame(data).to_json(filename, orient='records')
        else:
            raise ValueError("Unsupported format. Use 'csv' or 'json'")
            
        return filename

if __name__ == "__main__":
    orchestrator = DatasetOrchestrator()
    
    user_query = '''
    Generate 200 rows of Zomato customer dataset with columns:
    - "id" (number range: 1000-9999)
    - "name" (text)
    - "age" (number range: 18-90)
    - "city" (category options: New York, London, Tokyo, Paris)
    - "review" (text)
    '''
    dataset_path = orchestrator.generate_dataset(
        query=user_query,
        num_rows=50,
        format='csv'
    )
    
    print(f"Dataset generated at: {dataset_path}")
    print(f"Sample data:\n{pd.read_csv(dataset_path).head()}")
