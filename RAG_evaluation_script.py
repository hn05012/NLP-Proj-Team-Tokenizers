import csv
import time
import psutil
import json
import os
import gc
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
import warnings
warnings.filterwarnings("ignore")

# Optimized Config for Kaggle GPU T4
RETRIEVER_MODELS = {
    #"MiniLM": "all-MiniLM-L6-v2",  # Lightweight, good performance
    "BGE-small": "BAAI/bge-small-en-v1.5",  # Good balance,
    "gte-base": "thenlper/gte-base"
}

GENERATOR_MODELS = {
    "flan-t5-small": "google/flan-t5-small",  # Best fit for T4 memory
    "flan-t5-base": "google/flan-t5-base",   # Will work with ,
    "flan-t5-large": "google/flan-t5-large",
    "deepseek-6.7": "deepseek-ai/deepseek-coder-6.7b-instruct"
}

# File paths
TEST_FILE = "/kaggle/input/test-inputs-2-mini-json/test_inputs_2_mini.json"  # Update this path
RETRIEVER_RESULTS = "/kaggle/working/retriever_results.csv"
GENERATOR_RESULTS = "/kaggle/working/generator_results_kaggle.csv"

# GPU Memory optimization settings
MAX_BATCH_SIZE = 8  # For T4's 16GB memory
TORCH_COMPILE = False  # Set to True if using PyTorch 2.0+

def setup_gpu_optimization():
    """Setup GPU optimizations for Kaggle T4"""
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache
        torch.cuda.empty_cache()
    else:
        print("CUDA not available, using CPU")

def print_memory_usage():
    """Print detailed memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    print(f"\nMemory Usage:")
    print(f"  Process RSS: {mem_info.rss / (1024 ** 2):.2f} MB")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"  GPU Total: {gpu_memory:.2f} GB")
        print(f"  GPU Allocated: {gpu_allocated:.2f} GB")
        print(f"  GPU Reserved: {gpu_reserved:.2f} GB")

class OptimizedRetriever:
    """Optimized retriever for Kaggle environment"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        print(f"Loading retriever: {model_name}")
        
        # Load with optimizations
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print(f"Retriever loaded successfully")
        print_memory_usage()
    
    def encode_batch(self, texts: List[str], batch_size: int = MAX_BATCH_SIZE):
        """Encode texts in batches to manage memory"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                embeddings.append(batch_embeddings.cpu().numpy())
                
                # Clear GPU cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return np.vstack(embeddings)

class OptimizedGenerator:
    """Optimized generator for Kaggle T4 GPU"""

    def __init__(self, model_name: str = 'google/flan-t5-small'):
        self.model_name = model_name
        print(f"Loading generator: {model_name}")
        print_memory_usage()

        try:
            # Clear memory before loading
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load config first to determine model type
            config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            # Auto-detect model type
            if config.is_encoder_decoder:
                print("Detected encoder-decoder model (Seq2Seq).")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.is_seq2seq = True
            else:
                print("Detected decoder-only model (Causal LM).")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.is_seq2seq = False

            self.model.eval()
            self.device = next(self.model.parameters()).device
            print(f"Generator loaded on device: {self.device}")
            print_memory_usage()

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise  # Optional: fail fast instead of falling back to CPU silently
    
    # def __init__(self, model_name: str = 'google/flan-t5-small'):
    #     self.model_name = model_name
    #     print(f"Loading generator: {model_name}")
    #     print_memory_usage()
        
    #     try:
    #         # Clear memory before loading
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
            
    #         # Load tokenizer
    #         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
    #         # Load model with memory optimizations
    #         self.model = AutoModelForSeq2SeqLM.from_pretrained(
    #             model_name,
    #             torch_dtype=torch.float16,  # Use half precision
    #             low_cpu_mem_usage=True,
    #             device_map="auto" if torch.cuda.is_available() else None,
    #         )
            
    #         self.model.eval()
    #         self.device = next(self.model.parameters()).device
            
    #         # Apply optimizations
    #         if hasattr(self.model, 'gradient_checkpointing_enable'):
    #             self.model.gradient_checkpointing_enable()
            
    #         print(f"Generator loaded on device: {self.device}")
    #         print_memory_usage()
            
    #     except Exception as e:
    #         print(f"Error loading model: {str(e)}")
    #         # Fallback to CPU if GPU fails
    #         if torch.cuda.is_available():
    #             print("Trying CPU fallback...")
    #             self.model = AutoModelForSeq2SeqLM.from_pretrained(
    #                 model_name,
    #                 torch_dtype=torch.float32,
    #                 low_cpu_mem_usage=True,
    #             )
    #             self.device = torch.device('cpu')
    #             self.model.to(self.device)
    
    def build_prompt(self, question: str, context_chunks: List[Dict[str, Any]], 
                 max_context_length: int = 800) -> str:
        """Build prompt based on model type (seq2seq vs decoder-only)"""
        
        # Determine formatting style from model config
        is_decoder_only = not getattr(self.model.config, "is_encoder_decoder", False)
    
        # Sort context chunks by relevance
        sorted_chunks = sorted(context_chunks, key=lambda x: x['score'], reverse=True)
        context_parts = []
        current_length = 0
    
        for chunk_info in sorted_chunks:
            chunk_text = chunk_info['chunk']
            chunk_with_source = f"[Source: {chunk_info['source']}] {chunk_text}"
    
            if current_length + len(chunk_with_source) <= max_context_length:
                context_parts.append(chunk_with_source)
                current_length += len(chunk_with_source)
            else:
                remaining_space = max_context_length - current_length
                if remaining_space > 50:
                    context_parts.append(chunk_with_source[:remaining_space] + "...")
                break
    
        context = "\n\n".join(context_parts)
    
        # Build the prompt based on the model type
        if is_decoder_only:
            # Decoder-only format (no explicit "Answer the question" prompt)
            prompt = f"{context}\n\nQ: {question}\nA:"
        else:
            # Seq2seq style prompt
            prompt = f"""Answer the question based on the context. Be concise.

Context:
{context}

Question: {question}
"""

        return prompt

    
    def generate_answer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate answer with memory management"""
        try:
            # Tokenize with length limits
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=1024,  # Limit input length
                truncation=True
            ).to(self.device)
            
            generation_kwargs = {
                **kwargs,
                'max_new_tokens': 500,  # Limit output length
                'num_beams': 2,  # Reduce beam search
                'do_sample': True,
                'temperature': 0.7,
                'pad_token_id': self.tokenizer.eos_token_id,
                **kwargs
            }
            
            with torch.no_grad():
                if self.is_seq2seq:
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                else:
                    outputs = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        **generation_kwargs
                    )
            
            # Decode answer
            answer = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Simple confidence estimation
            confidence = min(0.8, len(answer.split()) / 20)  # Rough heuristic
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'answer': answer,
                'confidence': round(confidence, 3),
                'is_fallback': confidence < 0.3
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU OOM - trying with reduced parameters")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Retry with more conservative settings
                return self.generate_answer(
                    prompt, 
                    max_new_tokens=500, 
                    num_beams=1,
                    **kwargs
                )
            else:
                raise
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]], 
                       **generation_kwargs) -> Dict[str, Any]:
        """Answer question with context"""
        prompt = self.build_prompt(question, context_chunks)
        answer = self.generate_answer(prompt, **generation_kwargs)
        
        return {
            'question': question,
            'answer': answer,
            'num_context_chunks': len(context_chunks),
            'context_sources': [chunk['source'] for chunk in context_chunks]
        }

def load_test_data(filepath):
    """Load and validate test data"""
    try:
        with open(filepath) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Test file not found: {filepath}")
        print("Creating sample test data...")
        # Create sample data if file doesn't exist
        data = [
            {
                "question": "What is Python?",
                "expected_answer": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "context_chunks": [
                    {"text": "Python is a high-level programming language that emphasizes code readability.", "contains_answer": true, "score": 0.8},
                    {"text": "Unlike Java, Python uses dynamic typing and an interpreter instead of a compiler.", "contains_answer": false, "score": 0.1}
                ],
                "metadata": {
                    "tags": ["python", "programming"],
                    "question_score": 85,
                    "answer_score": 120,
                    "created": "2023-01-15T10:30:00",
                    "question_id": 12345678,
                    "answer_id": 23456789
                }
            },
            {
                "question": "How to fix 'ModuleNotFoundError: No module named numpy' in Python?",
                "expected_answer": "Install numpy using pip: pip install numpy",
                "context_chunks": [
                    {"text": "Common Python package errors can be resolved by checking pip installations.", "contains_answer": false, "score": 0.3},
                    {"text": "For missing numpy, run: pip install numpy", "contains_answer": true, "score": 1.0},
                    {"text": "Virtual environments can help manage package dependencies.", "contains_answer": false, "score": 0.2}
                ],
                "metadata": {
                    "tags": ["python", "numpy", "error"],
                    "question_score": 150,
                    "answer_score": 210,
                    "created": "2023-02-20T14:25:00",
                    "question_id": 34567890,
                    "answer_id": 45678901
                }
            },
            {
                "question": "What is the difference between lists and tuples in Python?",
                "expected_answer": "Lists are mutable (can be modified) while tuples are immutable (cannot be modified after creation).",
                "context_chunks": [
                    {"text": "Python lists use square brackets and can be modified.", "contains_answer": true, "score": 0.7},
                    {"text": "Tuples use parentheses and are immutable.", "contains_answer": true, "score": 0.7},
                    {"text": "Both lists and tuples can store multiple items.", "contains_answer": false, "score": 0.1}
                ],
                "metadata": {
                    "tags": ["python", "data-structures"],
                    "question_score": 95,
                    "answer_score": 135,
                    "created": "2023-03-10T09:15:00",
                    "question_id": 56789012,
                    "answer_id": 67890123
                }
            },
            {
                "question": "How to create a virtual environment in Python?",
                "expected_answer": "Use python -m venv env_name to create a virtual environment named 'env_name'.",
                "context_chunks": [
                    {"text": "Virtual environments isolate project dependencies.", "contains_answer": false, "score": 0.2},
                    {"text": "Run: python -m venv myenv to create a virtual environment.", "contains_answer": true, "score": 1.0},
                    {"text": "Activate with: source myenv/bin/activate (Linux/Mac)", "contains_answer": false, "score": 0.4}
                ],
                "metadata": {
                    "tags": ["python", "virtualenv"],
                    "question_score": 110,
                    "answer_score": 175,
                    "created": "2023-04-05T16:40:00",
                    "question_id": 78901234,
                    "answer_id": 89012345
                }
            },
            {
                "question": "What is PEP 8 in Python?",
                "expected_answer": "PEP 8 is Python's official style guide that provides coding conventions for readable code.",
                "context_chunks": [
                    {"text": "PEP 8 defines Python's style guidelines.", "contains_answer": true, "score": 0.9},
                    {"text": "It covers naming conventions, indentation, and line length.", "contains_answer": false, "score": 0.5},
                    {"text": "Other PEPs propose language features.", "contains_answer": false, "score": 0.1}
                ],
                "metadata": {
                    "tags": ["python", "coding-style"],
                    "question_score": 80,
                    "answer_score": 125,
                    "created": "2023-05-12T11:20:00",
                    "question_id": 90123456,
                    "answer_id": 1234567
                }
            },
            {
                "question": "How to handle exceptions in Python?",
                "expected_answer": "Use try-except blocks: try: risky_code() except ExceptionType: handle_error()",
                "context_chunks": [
                    {"text": "Python uses try-except for exception handling.", "contains_answer": true, "score": 0.8},
                    {"text": "You can catch specific exceptions or use a general Exception.", "contains_answer": false, "score": 0.4},
                    {"text": "Finally blocks execute code regardless of exceptions.", "contains_answer": false, "score": 0.3}
                ],
                "metadata": {
                    "tags": ["python", "error-handling"],
                    "question_score": 100,
                    "answer_score": 155,
                    "created": "2023-06-18T13:50:00",
                    "question_id": 2345678,
                    "answer_id": 3456789
                }
            },
            {
                "question": "What is the difference between == and is in Python?",
                "expected_answer": "== compares values while is checks if two variables reference the same object in memory.",
                "context_chunks": [
                    {"text": "The == operator compares values for equality.", "contains_answer": true, "score": 0.7},
                    {"text": "The is operator checks for identity (same memory location).", "contains_answer": true, "score": 0.7},
                    {"text": "For small integers, Python may cache objects.", "contains_answer": false, "score": 0.2}
                ],
                "metadata": {
                    "tags": ["python", "operators"],
                    "question_score": 90,
                    "answer_score": 140,
                    "created": "2023-07-22T08:10:00",
                    "question_id": 4567890,
                    "answer_id": 5678901
                }
            }
        ]
    
    # Ensure all items have required fields
    for item in data:
        if "context_chunks" not in item:
            item["context_chunks"] = [
                {"text": item.get("expected_answer", "Sample answer"), "contains_answer": True},
                {"text": "This is irrelevant context.", "contains_answer": False}
            ]
    
    return data

def benchmark_retriever(model_name, test_data):
    """Benchmark retriever performance"""
    print(f"\nBenchmarking retriever: {model_name}")
    print_memory_usage()
    
    try:
        retriever = OptimizedRetriever(model_name)
        results = []
        
        for i, query in enumerate(test_data):
            print(f"Processing query {i+1}/{len(test_data)}")
            
            docs = [chunk["text"] for chunk in query["context_chunks"]]
            gold_chunks = [i for i, chunk in enumerate(query["context_chunks"]) 
                          if chunk["contains_answer"]]
            
            start_time = time.perf_counter()
            
            # Encode documents and query
            doc_embeddings = retriever.encode_batch(docs)
            query_embedding = retriever.encode_batch([query["question"]])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            latency = (time.perf_counter() - start_time) * 1000
            
            # Calculate metrics
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Calculate ranking metrics
            hit_at_1 = int(any(i in sorted_indices[:1] for i in gold_chunks))
            hit_at_3 = int(any(i in sorted_indices[:3] for i in gold_chunks))
            hit_at_5 = int(any(i in sorted_indices[:5] for i in gold_chunks))
            
            # MRR calculation
            mrr = 0
            if gold_chunks:
                for gold_idx in gold_chunks:
                    rank_position = np.where(sorted_indices == gold_idx)[0]
                    if len(rank_position) > 0:
                        mrr = max(mrr, 1.0 / (rank_position[0] + 1))
            
            results.append({
                "model": model_name,
                "question": query["question"][:100],  # Truncate for CSV
                "hit@1": hit_at_1,
                "hit@3": hit_at_3,
                "hit@5": hit_at_5,
                "mrr": round(mrr, 4),
                "latency_ms": round(latency, 2),
                "memory_mb": round(psutil.Process().memory_info().rss / 1024**2, 1)
            })
            
            # Memory cleanup
            if i % 5 == 0:  # Every 5 queries
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"Error in retriever benchmark: {str(e)}")
        return []

def benchmark_generator(model_name, test_data):
    """Benchmark generator performance"""
    print(f"\nBenchmarking generator: {model_name}")
    print_memory_usage()
    
    try:
        generator = OptimizedGenerator(model_name=model_name)
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        results = []
        
        for i, item in enumerate(test_data):
            print(f"Processing item {i+1}/{len(test_data)}")
            
            # Prepare context chunks
            context_chunks = [
                {
                    "chunk": chunk["text"], 
                    "source": f"chunk_{j}", 
                    "score": 1.0 if chunk["contains_answer"] else 0.1
                }
                for j, chunk in enumerate(item["context_chunks"])
            ]
            
            start_time = time.perf_counter()
            
            # Generate answer
            output = generator.answer_question(
                question=item["question"],
                context_chunks=context_chunks,
                max_new_tokens=500  # Conservative for T4
            )
            
            latency = (time.perf_counter() - start_time) * 1000
            
            # Calculate metrics
            generated_answer = output["answer"]["answer"]
            expected_answer = item["expected_answer"]
            
            # ROUGE score
            rouge_scores = rouge_scorer_obj.score(expected_answer, generated_answer)
            rouge_l = rouge_scores["rougeL"].fmeasure
            
            # BERT score (with error handling)
            try:
                bert_scores = bert_score(
                    [generated_answer], 
                    [expected_answer], 
                    lang="en", 
                    verbose=False
                )
                bert_f1 = bert_scores[2].mean().item()  # F1 score
            except Exception as e:
                print(f"BERT score error: {e}")
                bert_f1 = 0.0
            
            # Hallucination detection
            is_hallucination = (
                "cannot find" not in generated_answer.lower() and 
                "cannot answer" not in generated_answer.lower() and
                not any(chunk["contains_answer"] for chunk in item["context_chunks"])
            )
            
            results.append({
                "model": model_name,
                "question": item["question"][:100],
                "expected_answer": expected_answer,
                # "expected_answer": expected_answer[:200],
                # "generated_answer": generated_answer[:200],
                "generated_answer": generated_answer,
                "confidence": output["answer"]["confidence"],
                "is_fallback": output["answer"]["is_fallback"],
                "rougeL": round(rouge_l, 4),
                "bert_score": round(bert_f1, 4),
                "latency_ms": round(latency, 2),
                "is_hallucination": int(is_hallucination),
                "memory_mb": round(psutil.Process().memory_info().rss / 1024**2, 1)
            })
            
            # Memory cleanup
            if i % 3 == 0:  # Every 3 items for generators
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return results
        
    except Exception as e:
        print(f"Error in generator benchmark: {str(e)}")
        return []

def save_results(results, filepath):
    """Save results to CSV"""
    if not results:
        print("No results to save")
        return
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {filepath}")

def main():
    """Main benchmark execution"""
    print("="*60)
    print("RAG BENCHMARK - Optimized for Kaggle GPU T4")
    print("="*60)
    
    # Setup GPU optimizations
    setup_gpu_optimization()
    print_memory_usage()
    
    try:
        # Load test data
        test_data = load_test_data(TEST_FILE)
        print(f"Loaded {len(test_data)} test items")
        
        # Benchmark retrievers
        print("\n" + "="*40)
        print("BENCHMARKING RETRIEVERS")
        print("="*40)
        
        all_retriever_results = []
        for model_name, model_path in RETRIEVER_MODELS.items():
            try:
                results = benchmark_retriever(model_path, test_data)
                all_retriever_results.extend(results)
                
                # Cleanup between models
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Failed to benchmark retriever {model_name}: {str(e)}")
                continue
        
        if all_retriever_results:
            save_results(all_retriever_results, RETRIEVER_RESULTS)
        
        # Benchmark generators
        print("\n" + "="*40) 
        print("BENCHMARKING GENERATORS")
        print("="*40)
        
        all_generator_results = []
        for model_name, model_path in GENERATOR_MODELS.items():
            try:
                results = benchmark_generator(model_path, test_data)
                all_generator_results.extend(results)
                
                # Cleanup between models
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Failed to benchmark generator {model_name}: {str(e)}")
                continue
        
        if all_generator_results:
            save_results(all_generator_results, GENERATOR_RESULTS)
        
        print("\n" + "="*40)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*40)
        
    except Exception as e:
        print(f"Main process error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\nFinal memory usage:")
        print_memory_usage()

if __name__ == "__main__":
    main()