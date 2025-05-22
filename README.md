# RAG Pipeline - Baseline Implementation

This project implements a Retrieval-Augmented Generation (RAG) system that combines document retrieval with text generation to answer questions based on provided context documents.

## Project Structure

```
├── data/
│   ├── test_inputs.json          # Test questions and expected answers
│   └── [your documents here]     # .txt, .md, or .pdf files
├── retriever/
│   └── retriever.py              # Retriever class implementation
├── generator/
│   └── generator.py              # Generator class implementation
├── logs/                         # Generated log files
│   ├── rag_logs.jsonl           # Query logs in JSONL format
│   └── test_results_*.json      # Test results
├── pipeline.py                   # Main pipeline orchestration
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## Features

### Retriever Class (`retriever/retriever.py`)

- **Document Processing**: Supports `.txt`, `.md`, and `.pdf` files
- **Text Chunking**: Intelligent chunking with overlap to maintain context
- **Embeddings**: Uses SentenceTransformers (default: `all-MiniLM-L6-v2`)
- **Indexing**: FAISS-based vector indexing for fast similarity search
- **Persistence**: Save/load functionality for indexes and embeddings

**Key Methods:**

- `add_documents(documents)`: Add documents (file paths or raw text)
- `query(question, k=5)`: Retrieve top-k relevant chunks
- `save(filepath)`: Save retriever state to disk
- `load(filepath)`: Load retriever state from disk

### Generator Class (`generator/generator.py`)

- **Model Integration**: Uses HuggingFace's T5 models (default: `google/flan-t5-small`)
- **Prompt Engineering**: Dynamically constructs prompts with retrieved context
- **Generation Control**: Configurable parameters (temperature, beam search, etc.)
- **Fallback Handling**: Identifies when answers cannot be found in context

**Key Methods:**

- `build_prompt(question, context_chunks)`: Creates context-aware prompts
- `generate_answer(prompt)`: Generates answers using the LLM
- `answer_question(question, context_chunks)`: End-to-end question answering
