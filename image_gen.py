import os
import re
import csv
import json
import random
from typing import Dict, List, Tuple
import pandas as pd
from PIL import Image
import numpy as np
import requests
from io import BytesIO


from diffusers import StableDiffusionPipeline
import torch

TEMP_STORAGE = "generated_datasets"
os.makedirs(TEMP_STORAGE, exist_ok=True)

class DatasetSchema:
    def __init__(self, query: str):
        self.columns = self._parse_query(query)
        
    def _parse_query(self, query: str) -> List[Dict]:
        columns = []
        patterns = [
            r'"(\w+)"\s*\((\w+)(?:\s*resolution:\s*([\d\-x]+))?(?:\s*options:\s*([\w\s,]+))?\)',
            r'column\s+(\w+)\s+as\s+(\w+)(?:\s+with\s+(.*))?'
        ]
        
        for match in re.finditer(patterns[0], query):
            name, dtype = match.group(1), match.group(2).lower()
            col_def = {'name': name, 'type': dtype}
            
            if dtype == 'image':
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

class ImageDataGenerator:
    def __init__(self):
        self.model = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
    def generate_image(self, prompt: str, width: int = 512, height: int = 512) -> Image.Image:
        """Generate an image based on the prompt using Stable Diffusion."""
        try:
            result = self.model(prompt, width=width, height=height)
            return result.images[0]
        except Exception as e:
            print(f"Error generating image: {e}")
            return Image.new('RGB', (width, height), color = (73, 109, 137))
    
    @staticmethod
    def generate_category(col_def: Dict) -> str:
        return random.choice(col_def.get('options', ['Unknown']))

class DatasetOrchestrator:
    def __init__(self):
        self.image_gen = ImageDataGenerator()
        
    def generate_dataset(self, query: str, num_rows: int = 10, format: str = 'folder') -> str:
        schema = DatasetSchema(query)
        data = []
        
        prompts_pattern = r'prompts:\s*\[(.*?)\]'
        prompts_match = re.search(prompts_pattern, query)
        
        if prompts_match:
            prompts_str = prompts_match.group(1)
            prompts = [p.strip(' "\'') for p in prompts_str.split(',')]
        else:
            prompts = ["a photo of a cat", "a landscape photo", "portrait of a person"]
        
        dataset_id = f"image_dataset_{hash(str(schema.columns))}"
        dataset_path = os.path.join(TEMP_STORAGE, dataset_id)
        os.makedirs(dataset_path, exist_ok=True)
        
        metadata = {
            "dataset_id": dataset_id,
            "num_images": num_rows,
            "schema": [col for col in schema.columns],
            "prompts": prompts,
            "images": []
        }
        
        print(f"Generating {num_rows} images...")
        
        for i in range(num_rows):
            row = {}
            image_metadata = {}
            
            prompt = random.choice(prompts)
            
            for col in schema.columns:
                if col['type'] == 'image':
                    width = col.get('width', 512)
                    height = col.get('height', 512)
                    
                    custom_prompt = f"{prompt}, high quality, detailed"
                    
                    image = self.image_gen.generate_image(custom_prompt, width, height)
                    
                    image_filename = f"{i:04d}_{col['name']}.png"
                    image_path = os.path.join(dataset_path, image_filename)
                    image.save(image_path)
                    
                    row[col['name']] = image_path
                    image_metadata[col['name']] = {
                        "path": image_path,
                        "prompt": custom_prompt,
                        "width": width,
                        "height": height
                    }
                    
                elif col['type'] == 'category':
                    value = self.image_gen.generate_category(col)
                    row[col['name']] = value
                    image_metadata[col['name']] = value
            
            data.append(row)
            metadata["images"].append(image_metadata)
            
            if (i + 1) % 5 == 0 or i == num_rows - 1:
                print(f"Generated {i + 1}/{num_rows} images")
        
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        csv_path = os.path.join(dataset_path, "index.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        if format == 'zip':
            import shutil
            zip_path = f"{dataset_path}.zip"
            shutil.make_archive(dataset_path, 'zip', dataset_path)
            return zip_path
            
        return dataset_path

if __name__ == "__main__":
    orchestrator = DatasetOrchestrator()
    
    user_query = '''
    Generate 10 images of dataset with columns:
    - "main_image" (image resolution: 512x512)
    - "style" (category options: realistic, cartoon, sketch, painting)
    - "background" (category options: indoor, outdoor, studio, abstract)
    prompts: ["a dog playing in a park", "a cat sleeping on a couch", "a bird on a tree"]
    '''
    
    dataset_path = orchestrator.generate_dataset(
        query=user_query,
        num_rows=10,
        format='folder'
    )
    
    print(f"Image dataset generated at: {dataset_path}")
    print(f"A total of 10 images were created with metadata")
