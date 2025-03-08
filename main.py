import os
import re
import argparse
import pandas as pd
from typing import Dict, List, Union

from text_dataset_generator import DatasetOrchestrator as TextDatasetOrchestrator
from image_dataset_generator import DatasetOrchestrator as ImageDatasetOrchestrator
from dataset_search_agent import DatasetSearchAgent

TEMP_STORAGE = "generated_datasets"
os.makedirs(TEMP_STORAGE, exist_ok=True)

class DatasetRouterAgent:    
    def __init__(self):
        self.text_orchestrator = TextDatasetOrchestrator()
        self.image_orchestrator = ImageDatasetOrchestrator()
        self.search_agent = DatasetSearchAgent()
    
    def process_query(self, query: str) -> str:
        if re.search(r'\b(search|find|look\s+for|existing)\b', query.lower()):
            return self._handle_search_query(query) 
 
        download_match = re.search(r'download\s+(?:dataset)?\s*(\d+)', query.lower())
        if download_match:
            return self._handle_download_request(query, download_match)
        
        if re.search(r'\b(generate|create|make|produce)\b', query.lower()):
            if re.search(r'\b(image|picture|photo)\b', query.lower()):
                return self._handle_image_generation(query)
            else:
                return self._handle_text_generation(query)
        
        return self._provide_help()

    def _handle_search_query(self, query: str) -> str:
        search_terms = re.sub(r'\b(search|find|look\s+for|existing)\s+(datasets?|data)?\s*(about|for|related\s+to)?\s*', '', query.lower())
        search_terms = search_terms.strip('?., ')
        
        print(f"Searching for datasets related to '{search_terms}'...")
        results = self.search_agent.search_datasets(search_terms)
        
        if results.empty:
            return "No datasets found matching your query."
        else:
            result_str = f"Found {len(results)} datasets matching your query:\n\n"
            for i, (_, row) in enumerate(results.iterrows()):
                result_str += f"{i}. {row['Title']}\n   {row['Description']}\n   Format: {row['Format']}, Size: {row['Size']}, Source: {row['Source']}\n\n"
            
            result_str += "To download one of these datasets, please specify its number."
            return result_str

    def _handle_download_request(self, query: str, download_match) -> str:
        """Handle requests to download a specific dataset"""
        dataset_index = int(download_match.group(1))
        results = self.search_agent.sample_catalog         
        if dataset_index < 0 or dataset_index >= len(results):
            return f"Invalid dataset number. Please select a number between 0 and {len(results)-1}."
            
        selected_dataset = results.iloc[dataset_index]
        dataset_path = self.search_agent.download_dataset(selected_dataset["Download Link"])
        
        return f"Dataset '{selected_dataset['Title']}' downloaded successfully to {dataset_path}"

    def _handle_image_generation(self, query: str) -> str:
        num_rows_match = re.search(r'(\d+)\s+(images|rows)', query)
        num_rows = int(num_rows_match.group(1)) if num_rows_match else 10
        
        format_match = re.search(r'format:\s*(\w+)', query.lower())
        format = format_match.group(1) if format_match else 'folder'
        
        try:
            dataset_path = self.image_orchestrator.generate_dataset(
                query=query,
                num_rows=num_rows,
                format=format
            )
            return f"Image dataset generated successfully at {dataset_path}"
        except Exception as e:
            return f"Error generating image dataset: {str(e)}"

    def _handle_text_generation(self, query: str) -> str:
        num_rows_match = re.search(r'(\d+)\s+(rows)', query)
        num_rows = int(num_rows_match.group(1)) if num_rows_match else 100
        
        format_match = re.search(r'format:\s*(\w+)', query.lower())
        format = format_match.group(1) if format_match else 'csv'
        
        try:
            dataset_path = self.text_orchestrator.generate_dataset(
                query=query,
                num_rows=num_rows,
                format=format
            )
            if format == 'csv':
                sample_data = pd.read_csv(dataset_path).head()
            elif format == 'json':
                sample_data = pd.read_json(dataset_path).head()
            else:
                sample_data = "Sample data not available for this format."
            
            return f"Text dataset generated successfully at {dataset_path}\n\nSample data:\n{sample_data}"
        except Exception as e:
            return f"Error generating text dataset: {str(e)}"

    def _provide_help(self) -> str:
        """Provide help information when the intent can't be determined"""
        return (
            "I'm a dataset agent that can help you with the following tasks:\n\n"
            "1. Search for existing datasets: 'Search for datasets about climate change'\n"
            "2. Generate text datasets: 'Generate 200 rows of customer data with columns \"id\" (number), \"name\" (text), \"age\" (number)'\n"
            "3. Generate image datasets: 'Generate 10 images of dogs with columns \"image\" (image), \"breed\" (category)'\n\n"
            "Please let me know what you'd like to do."
        )

def main():
    parser = argparse.ArgumentParser(description='Dataset Router Agent')
    parser.add_argument('--mode', type=str, choices=['interactive', 'cli'], default='interactive',
                       help='Run in interactive mode or with command-line arguments')
    parser.add_argument('--query', type=str, help='Query string for CLI mode')
    args = parser.parse_args()
    
    router = DatasetRouterAgent()
    
    if args.mode == 'interactive':
        print("=== Dataset Router Agent ===")
        print("You can search for datasets, generate text datasets, or generate image datasets.")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 30)
        
        while True:
            user_input = input("\nWhat would you like to do? > ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            response = router.process_query(user_input)
            print("\n" + response)
    else:
        # CLI mode
        if not args.query:
            print("Error: In CLI mode, you must provide a query with --query")
            return
            
        response = router.process_query(args.query)
        print(response)

if __name__ == "__main__":
    main()
