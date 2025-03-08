import os
import re
import argparse
from typing import Dict, List, Union, Optional, Any
from langchain_community.document_loaders import hugging_face_dataset
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import WikipediaRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HunggingFaceHub

from image_generator import ImageDatasetGenerator
from text_generator import TextDatasetGenerator
from dataset_searcher import DatasetSearcher

TEMP_STORAGE = "generated_datasets"
os.makedirs(TEMP_STORAGE, exist_ok=True)

class DatasetSchemaParser:
    
    @staticmethod
    def parse_query(query: str) -> List[Dict]:
        columns = []
        patterns = [
            r'"(\w+)"\s*\((\w+)(?:\s*(?:range|resolution):\s*([\d\-x]+))?(?:\s*options:\s*([\w\s,]+))?\)',
            r'column\s+(\w+)\s+as\s+(\w+)(?:\s+with\s+(.*))?'
        ]
        
        for match in re.finditer(patterns[0], query):
            name, dtype = match.group(1), match.group(2).lower()
            col_def = {'name': name, 'type': dtype}
            
            if dtype == 'number':
                if match.group(3):
                    min_val, max_val = map(int, match.group(3).split('-'))
                    col_def.update({'min': min_val, 'max': max_val})
            elif dtype == 'image':
                if match.group(3):
                    resolution = match.group(3)
                    width, height = map(int, resolution.split('x'))
                    col_def.update({'width': width, 'height': height})
                else:
                    col_def.update({'width': 512, 'height': 512})
            elif dtype == 'category' and match.group(4):
                options = [x.strip() for x in match.group(4).split(',')]
                col_def['options'] = options
                
            columns.append(col_def)
            
        return columns
    
    @staticmethod
    def extract_num_rows(query: str) -> int:
        patterns = [
            r'(\d+)\s+(rows|images)',
            r'generate\s+(\d+)',
            r'create\s+(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return 100 if 'image' not in query.lower() else 10
    
    @staticmethod
    def extract_format(query: str) -> str:
        format_match = re.search(r'format:\s*(\w+)', query.lower())
        format_type = format_match.group(1) if format_match else None
        
        if not format_type:
            if 'image' in query.lower():
                return 'folder'
            else:
                return 'csv'
        
        return format_type
    
    @staticmethod
    def extract_prompts(query: str) -> List[str]:
        prompts_pattern = r'prompts:\s*\[(.*?)\]'
        prompts_match = re.search(prompts_pattern, query)
        
        if prompts_match:
            prompts_str = prompts_match.group(1)
            return [p.strip(' "\'') for p in prompts_str.split(',')]
        
        return ["a photo of a cat", "a landscape photo", "portrait of a person"]

class DatasetRAG:
    
    def __init__(self, huggingface_api_token: Optional[str] = None):
        self.embeddings = None
        self.wiki_retriever = WikipediaRetriever()
        self.vectorstore = None
        self.retriever = None
        self.huggingface_api_token = huggingface_api_token
        
    def initialize_from_documents(self, documents_path: str):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        
        if documents_path.endswith('.csv'):
            loader = CSVLoader(documents_path)
        else:
            loader = TextLoader(documents_path)
            
        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)
        
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()
    
    def retrieve_context(self, query: str, sources: List[str] = None, k: int = 3) -> str:
        contexts = []
        
        if not sources or 'wikipedia' in sources:
            wiki_docs = self.wiki_retriever.get_relevant_documents(query)
            if wiki_docs:
                contexts.extend([doc.page_content for doc in wiki_docs[:k]])
        
        if self.retriever and (not sources or 'local' in sources):
            docs = self.retriever.get_relevant_documents(query)
            if docs:
                contexts.extend([doc.page_content for doc in docs[:k]])
        
        return "\n\n".join(contexts)
    
    def enrich_dataset(self, data: List[Dict], query: str) -> List[Dict]:

        enriched_data = data.copy()
        
        enrichment_topic = re.sub(r'\b(generate|create|make|produce)\b.*\bdata(set)?\b', '', query, flags=re.IGNORECASE).strip()
        
        if not enrichment_topic:
            return data
        
        context = self.retrieve_context(enrichment_topic)
        if not context:
            return data
        
        prompt = PromptTemplate(
            input_variables=["context", "data_sample"],
            template="""
            Based on the following context:
            {context}
            
            And looking at this sample data:
            {data_sample}
            
            Generate 3 meaningful insights or additional data points that could enrich this dataset.
            Return the insights as a JSON array of objects, where each object has a key 'insight' and value is your insight.
            """
        )
        
        llm = OpenAI(temperature=0.7)
        chain = LLMChain(llm=llm, prompt=prompt)
        

        sample_data = str(enriched_data[:3])
        insights_text = chain.run(context=context, data_sample=sample_data)
        
        try:
            import json
            insights = json.loads(insights_text)
            for row in enriched_data:
                if 'metadata' not in row:
                    row['metadata'] = {}
                row['metadata']['insights'] = insights
        except:
            pass
        
        return enriched_data

class DatasetRouterAgent:

    
    def __init__(self, api_key: Optional[str] = None):
        self.text_generator = TextDatasetGenerator(huggingface_api_token=hugging_face_token)
        self.image_generator = ImageDatasetGenerator()
        self.dataset_searcher = DatasetSearcher()
        self.rag = DatasetRAG(api_key=api_key)
        

        self.initialize_agent()
    
    def initialize_agent(self):
        from langchain.llms import HunggingFaceHub

        llm = OpenAI(
            temperature=0,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )
        
        tools = [
            Tool(
                name="SearchDatasets",
                func=self._handle_search_query,
                description="Search for existing datasets when the user asks to find or search for datasets"
            ),
            Tool(
                name="DownloadDataset",
                func=self._handle_download_request,
                description="Download a dataset when the user specifies a dataset number to download"
            ),
            Tool(
                name="GenerateImageDataset",
                func=self._handle_image_generation,
                description="Generate image datasets when the user asks to create datasets with images"
            ),
            Tool(
                name="GenerateTextDataset",
                func=self._handle_text_generation,
                description="Generate text-based datasets when the user asks to create datasets with text, numbers, or categories"
            ),
            Tool(
                name="Help",
                func=self._provide_help,
                description="Provide help information when the user's intent is unclear or they ask for help"
            )
        ]
        
        memory = ConversationBufferMemory(memory_key="chat_history")
        
        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )
    
    def process_query(self, query: str) -> str:
        if re.search(r'\b(search|find|look\s+for|existing)\b', query.lower()):
            return self._handle_search_query(query) 
 
        download_match = re.search(r'download\s+(?:dataset)?\s*(\d+)', query.lower())
        if download_match:
            dataset_index = int(download_match.group(1))
            return self._handle_download_request(str(dataset_index))
        
        if re.search(r'\b(generate|create|make|produce)\b', query.lower()):
            if re.search(r'\b(image|picture|photo)\b', query.lower()):
                return self._handle_image_generation(query)
            else:
                return self._handle_text_generation(query)
        
        try:
            return self.agent.run(query)
        except Exception as e:
            print(f"Agent error: {e}")
            return self._provide_help()

    def _handle_search_query(self, query: str) -> str:
        search_terms = re.sub(r'\b(search|find|look\s+for|existing)\s+(datasets?|data)?\s*(about|for|related\s+to)?\s*', '', query.lower())
        search_terms = search_terms.strip('?., ')
        
        print(f"Searching for datasets related to '{search_terms}'...")
        results = self.dataset_searcher.search_datasets(search_terms)
        
        if results.empty:
            return "No datasets found matching your query."
        else:
            result_str = f"Found {len(results)} datasets matching your query:\n\n"
            for i, (_, row) in enumerate(results.iterrows()):
                result_str += f"{i}. {row['Title']}\n   {row['Description']}\n   Format: {row['Format']}, Size: {row['Size']}, Source: {row['Source']}\n\n"
            
            result_str += "To download one of these datasets, please specify its number."
            return result_str

    def _handle_download_request(self, dataset_index_str: str) -> str:
        try:
            dataset_index = int(dataset_index_str)
            results = self.dataset_searcher.get_recent_results()
            
            if results.empty:
                return "Please search for datasets first before trying to download."
                
            if dataset_index < 0 or dataset_index >= len(results):
                return f"Invalid dataset number. Please select a number between 0 and {len(results)-1}."
                
            selected_dataset = results.iloc[dataset_index]
            dataset_path = self.dataset_searcher.download_dataset(selected_dataset["Download Link"])
            
            return f"Dataset '{selected_dataset['Title']}' downloaded successfully to {dataset_path}"
        except ValueError:
            return "Invalid dataset number. Please provide a valid number."
        except Exception as e:
            return f"Error downloading dataset: {str(e)}"

    def _handle_image_generation(self, query: str) -> str:
        schema = DatasetSchemaParser.parse_query(query)
        num_rows = DatasetSchemaParser.extract_num_rows(query)
        format_type = DatasetSchemaParser.extract_format(query)
        prompts = DatasetSchemaParser.extract_prompts(query)
        
        try:
            dataset_path = self.image_generator.generate_dataset(
                schema=schema,
                num_rows=num_rows,
                format=format_type,
                prompts=prompts
            )
            
            if "enrich" in query.lower() or "enhance" in query.lower():
                metadata_path = os.path.join(dataset_path, "metadata.json")
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata['images'] = self.rag.enrich_dataset(metadata['images'], query)
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
            
            return f"Image dataset generated successfully at {dataset_path}"
        except Exception as e:
            return f"Error generating image dataset: {str(e)}"

    def _handle_text_generation(self, query: str) -> str:
        schema = DatasetSchemaParser.parse_query(query)
        num_rows = DatasetSchemaParser.extract_num_rows(query)
        format_type = DatasetSchemaParser.extract_format(query)
        
        try:
            dataset_path, data = self.text_generator.generate_dataset(
                schema=schema,
                num_rows=num_rows,
                format=format_type,
                with_data=True
            )
            

            if "enrich" in query.lower() or "enhance" in query.lower():
                enriched_data = self.rag.enrich_dataset(data, query)
                
                if format_type == 'csv':
                    pd.DataFrame(enriched_data).to_csv(dataset_path, index=False)
                elif format_type == 'json':
                    pd.DataFrame(enriched_data).to_json(dataset_path, orient='records')
            
            if format_type == 'csv':
                sample_data = pd.read_csv(dataset_path).head()
            elif format_type == 'json':
                sample_data = pd.read_json(dataset_path).head()
            else:
                sample_data = "Sample data not available for this format."
            
            return f"Text dataset generated successfully at {dataset_path}\n\nSample data:\n{sample_data}"
        except Exception as e:
            return f"Error generating text dataset: {str(e)}"

    def _provide_help(self) -> str:

        return (
            "I'm a dataset agent built with LangChain that can help you with the following tasks:\n\n"
            "1. Search for existing datasets: 'Search for datasets about climate change'\n"
            "2. Generate text datasets: 'Generate 200 rows of customer data with columns \"id\" (number), \"name\" (text), \"age\" (number)'\n"
            "3. Generate image datasets: 'Generate 10 images of dogs with columns \"image\" (image), \"breed\" (category)'\n"
            "4. Enrich datasets with RAG: 'Generate and enrich a dataset about climate change'\n\n"
            "Please let me know what you'd like to do."
        )

def main():
    parser = argparse.ArgumentParser(description='LangChain Dataset Router Agent')
    parser.add_argument('--mode', type=str, choices=['interactive', 'cli'], default='interactive',
                       help='Run in interactive mode or with command-line arguments')
    parser.add_argument('--query', type=str, help='Query string for CLI mode')
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OpenAI API key provided. Please set OPENAI_API_KEY environment variable or provide with --api-key.")
    
    router = DatasetRouterAgent(api_key=api_key)
    
    if args.mode == 'interactive':
        print("=== LangChain Dataset Router Agent ===")
        print("You can search for datasets, generate text datasets, or generate image datasets.")
        print("You can also request dataset enrichment using RAG.")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 40)
        
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
