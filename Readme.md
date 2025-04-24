# 🌟 Content Generation & Memory System

A powerful system for generating text datasets, creating images, and remembering your past generations for future reference! This framework leverages LangGraph and Gemini to provide a seamless experience for content creation.

## ✨ Features

### 🧠 Intelligent Query Routing
- Automatically detects if you want to generate text, images, or access your memory
- Uses RAG (Retrieval-Augmented Generation) to enhance your queries with relevant knowledge

### 📊 Text Data Generation
- Creates structured datasets based on your description
- Generates detailed schemas with field definitions
- Produces sample data that matches your specifications
- Validates the output against the schema

### 🖼️ Image Generation
- Creates images using Stable Diffusion XL
- Intelligently crafts prompts for your requested visuals
- Generates multiple variations of your concept

### 💾 Memory System
- Stores all your generated content for future reference
- Uses semantic search to find relevant past creations
- Provides detailed summaries of your previous work

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: langraph, sentence-transformers, chromadb, google-generativeai

### Installation

```bash
# Clone the repository
git clone https://github.com/arjuuuuunnnnn/GenAI_Syn_Dataset_Gen.git
cd GenAI_Syn_Dataset_Gen

# Install dependencies
pip install -r requirements.txt

# Create environment file
touch .env
```

Add your API keys to the `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
HF_API_TOKEN=your_huggingface_token_here
```

### Running the System

```bash
python main.py
```

## 💫 Example Usage

### Generate a Text Dataset
```
Enter your request: Create a dataset of 10 fictional space colonies with name, population, founding year, and primary export
```

### Generate Images
```
Enter your request: Generate images of a cyberpunk city with neon lights and flying cars
```

### Query Your Memory
```
Enter your request: What datasets did I create recently?
```

```
Enter your request: Tell me more about the space colonies dataset from earlier
```

## 🏗️ System Architecture

The system consists of the following components:

- **main.py**: Central router that analyzes queries and directs to appropriate agents
- **text_gen.py**: Specialized agent for text and dataset generation
- **image_gen.py**: Specialized agent for image creation
- **data_memory.py**: Storage system for generated content
- **memory_rag.py**: Enhanced RAG system for knowledge and memory integration

## 📝 Notes

- The memory system creates persistent storage in `./memory_db`
- Generated images are saved in `./generated_images`
- Dataset outputs are saved to JSON files for easy access

