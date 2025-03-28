import os
import random
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image
import numpy as np
import torch

try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers package not available. Install with: pip install diffusers")

TEMP_STORAGE = "generated_datasets"
os.makedirs(TEMP_STORAGE, exist_ok=True)

class ImageDatasetGenerator:    
    def __init__(self):
        self.model = None
        if DIFFUSERS_AVAILABLE:
            try:
                self.model = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5", 
                    torch_dtype=torch.float16
                )
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
            except Exception as e:
                print(f"Error initializing Stable Diffusion: {e}")
                print("Will use placeholder images instead.")
    
    def generate_image(self, prompt: str, width: int = 512, height: int = 512) -> Image.Image:
        if self.model is None:
            # Return a placeholder image if the model is not available
            return self._generate_placeholder_image(width, height, prompt)
        
        try:
            result = self.model(prompt, width=width, height=height)
            return result.images[0]
        except Exception as e:
            print(f"Error generating image: {e}")
            return self._generate_placeholder_image(width, height, prompt)
    
    def _generate_placeholder_image(self, width: int, height: int, prompt: str) -> Image.Image:
        from PIL import Image, ImageDraw, ImageFont
        
        image = Image.new('RGB', (width, height), color=(73, 109, 137))
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        text = f"Placeholder Image\n\nPrompt: {prompt[:100]}"
        draw.text((10, 10), text, fill=(255, 255, 255), font=font)
        
        return image
    
    @staticmethod
    def generate_category(col_def: Dict) -> str:
        """Generate a random category from options"""
        return random.choice(col_def.get('options', ['Unknown']))
    
    def generate_dataset(self, schema: List[Dict], num_rows: int = 10, 
                        format: str = 'folder', prompts: List[str] = None) -> str:
        data = []
        
        if not prompts:
            prompts = ["a photo of a cat", "a landscape photo", "portrait of a person"]
        
        import hashlib
        schema_str = json.dumps(schema, sort_keys=True)
        dataset_id = hashlib.md5(schema_str.encode()).hexdigest()[:10]
        
        dataset_path = os.path.join(TEMP_STORAGE, f"image_dataset_{dataset_id}")
        os.makedirs(dataset_path, exist_ok=True)
        
        metadata = {
            "dataset_id": dataset_id,
            "num_images": num_rows,
            "schema": schema,
            "prompts": prompts,
            "images": []
        }
        
        print(f"Generating {num_rows} images...")
        
        for i in range(num_rows):
            row = {}
            image_metadata = {}
            
            prompt = random.choice(prompts)
            
            for col in schema:
                if col['type'] == 'image':
                    width = col.get('width', 512)
                    height = col.get('height', 512)
                    
                    custom_prompt = f"{prompt}, high quality, detailed"
                    
                    image = self.generate_image(custom_prompt, width, height)
                    
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
                    value = self.generate_category(col)
                    row[col['name']] = value
                    image_metadata[col['name']] = value
            
            data.append(row)
            metadata["images"].append(image_metadata)
            
            if (i + 1) % 5 == 0 or i == num_rows - 1:
                print(f"Generated {i + 1}/{num_rows} images")
        
        metadata_path = os.path.join(dataset_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)  # Fixed: Added indented block
        
        return dataset_path
