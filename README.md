# Python RAG Pipeline

A streamlined Retrieval-Augmented Generation system for Python technical questions.

## Key Features

- **Optimized Retriever**  
  - Dual-mode search (FAISS/cosine)  
  - Fine-tuned embeddings (Triplet Loss)  
  - Context-aware document processing  

- **Precise Generator**  
  - DeepSeek-Coder 6.7B model  
  - Strict Python-focused responses  
  - Context verification protocol  

- **End-to-End Pipeline**  
  - Simple `query()` interface  
  - Automatic result formatting  
  - Built-in benchmarking  

## Quick Start

```python
from pipeline import RAGPipeline

# Initialize with your models
rag = RAGPipeline(
    retriever_model='your-retriever-path',
    generator_model='deepseek-ai/deepseek-coder-6.7b-instruct'
)

# Load your dataset
rag.add_documents_from_dataset('your_data.json')

# Query the system
result = rag.query("How do I use Python's dataclasses?")
rag.print_formatted_result(result)
