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
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoModelForSeq2SeqLM

# Config
RETRIEVER_MODELS = {
    "MiniLM": "all-MiniLM-L6-v2",
    "BGE-small": "BAAI/bge-small-en-v1.5",
    "E5-base": "intfloat/e5-base-v2"
}

GENERATOR_MODELS = {
    # "flan-t5-base": "google/flan-t5-base",
    # "flan-t5-large": "google/flan-t5-large",
    "flan-t5-large": "google/flan-t5-small",
    # "phi-2": "microsoft/phi-2"  
}

TEST_FILE = "data/test_inputs.json"
RETRIEVER_RESULTS = "results/retriever_results.csv"
GENERATOR_RESULTS = "results/generator_results.csv"

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    total_mem = psutil.virtual_memory().total / (1024 ** 3)
    available_mem = psutil.virtual_memory().available / (1024 ** 3)
    print(f"\nMemory Usage:")
    print(f"  Process RSS: {mem_info.rss / (1024 ** 2):.2f} MB")
    print(f"  Process VMS: {mem_info.vms / (1024 ** 2):.2f} MB")
    print(f"  System Total: {total_mem:.2f} GB")
    print(f"  System Available: {available_mem:.2f} GB")
    print(f"  System Used: {(total_mem - available_mem):.2f} GB")

class Generator:
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        self.model_name = model_name
        print(f"Loading generator: {model_name}")
        print_memory_usage()
        
        try:
            # Clear memory before loading
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            print(f"Generator loaded on device: {self.device}")
            print_memory_usage()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def build_prompt(self, question: str, context_chunks: List[Dict[str, Any]], max_context_length: int = 1000) -> str:
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
                if remaining_space > 100:  
                    context_parts.append(chunk_with_source[:remaining_space] + "...")
                break
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Answer:"""
        return prompt
    
    def generate_answer(self, prompt: str, **kwargs) -> Dict[str, Any]:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_length=200,  # Reduced from default to save memory
                    **kwargs
                )
            
            scores = torch.stack(outputs.scores, dim=1)
            confidence = torch.mean(torch.softmax(scores, dim=-1).max(dim=-1).values).item()
            answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            if confidence < 0.3 or "cannot find" in answer.lower():
                answer = "I cannot confidently answer based on the provided context."
            
            return {
                'answer': answer,
                'confidence': round(confidence, 3),
                'is_fallback': confidence < 0.3
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Out of memory error caught, retrying with garbage collection")
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                return self.generate_answer(prompt, **kwargs)
            raise

    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]], 
                       **generation_kwargs) -> Dict[str, Any]:
        prompt = self.build_prompt(question, context_chunks)
        answer = self.generate_answer(prompt, **generation_kwargs)
        
        return {
            'question': question,
            'prompt': prompt,
            'answer': answer,
            'num_context_chunks': len(context_chunks),
            'context_sources': [chunk['source'] for chunk in context_chunks]
        }

def load_test_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    for item in data:
        if "context_chunks" not in item:
            item["context_chunks"] = [
                {"text": item["expected_answer"], "contains_answer": True},
                {"text": "Python is a programming language.", "contains_answer": False}
            ]
    return data

def benchmark_retriever(model_name, test_data):
    print(f"Benchmarking retriever: {model_name}")
    print_memory_usage()
    
    try:
        model = SentenceTransformer(model_name)
        results = []
        
        for query in test_data:
            gc.collect()
            docs = [chunk["text"] for chunk in query["context_chunks"]]
            gold_chunks = [i for i, chunk in enumerate(query["context_chunks"]) 
                          if chunk["contains_answer"]]
            
            start_time = time.perf_counter()
            doc_embeddings = model.encode(docs)
            query_embedding = model.encode([query["question"]])
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            latency = (time.perf_counter() - start_time) * 1000
            
            sorted_indices = np.argsort(similarities)[::-1]
            ranks = {
                "hit@1": int(any(i in sorted_indices[:1] for i in gold_chunks)),
                "hit@3": int(any(i in sorted_indices[:3] for i in gold_chunks)),
                "hit@5": int(any(i in sorted_indices[:5] for i in gold_chunks)),
                "mrr": 1/(np.where(sorted_indices == gold_chunks[0])[0][0] + 1) 
                       if gold_chunks else 0
            }
            
            results.append({
                "model": model_name,
                "question": query["question"],
                **ranks,
                "latency_ms": round(latency, 2),
                "memory_mb": round(psutil.Process().memory_info().rss / 1024**2, 1)
            })
        
        return results
    except Exception as e:
        print(f"Error in retriever benchmark: {str(e)}")
        raise

def benchmark_generator(model_name, test_data):
    print(f"Benchmarking generator: {model_name}")
    print_memory_usage()
    
    try:
        generator = Generator(model_name=model_name)
        rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        results = []
        for item in test_data:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            context_chunks = [
                {"chunk": chunk["text"], "source": f"chunk_{i}", "score": 1.0 if chunk["contains_answer"] else 0.1}
                for i, chunk in enumerate(item["context_chunks"])
            ]
            
            start_time = time.perf_counter()
            output = generator.answer_question(
                question=item["question"],
                context_chunks=context_chunks,
                max_new_tokens=100
            )
            latency = (time.perf_counter() - start_time) * 1000
            
            rouge = rouge_scorer_obj.score(item["expected_answer"], output["answer"]["answer"])
            bert_scores = bert_score([output["answer"]["answer"]], [item["expected_answer"]], lang="en")
            
            is_hallucination = (
                "cannot find" not in output["answer"]["answer"].lower() and 
                not any(chunk["contains_answer"] for chunk in item["context_chunks"])
            )
            
            results.append({
                "model": model_name,
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "generated_answer": output["answer"]["answer"],
                "confidence": output["answer"]["confidence"],
                "is_fallback": output["answer"]["is_fallback"],
                "rougeL": rouge["rougeL"].fmeasure,
                "bert_score": bert_scores[0].mean().item(),
                "latency_ms": round(latency, 2),
                "is_hallucination": int(is_hallucination),
                "memory_mb": round(psutil.Process().memory_info().rss / 1024**2, 1)
            })
            
            print(f"Processed question: {item['question'][:50]}...")
            print_memory_usage()
        
        return results
    except Exception as e:
        print(f"Error in generator benchmark: {str(e)}")
        raise

def save_results(results, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filepath}")

def main():
    print("Starting benchmark...")
    print_memory_usage()
    
    try:
        test_data = load_test_data(TEST_FILE)
        
        # Benchmark retrievers
        # print("\nBenchmarking retrievers...")
        # retriever_results = []
        # for model_name in RETRIEVER_MODELS.values():
        #     retriever_results.extend(benchmark_retriever(model_name, test_data))
        # save_results(retriever_results, RETRIEVER_RESULTS)
        
        # Benchmark generators
        print("\nBenchmarking generators...")
        generator_results = []
        for model_name in GENERATOR_MODELS.values():
            try:
                generator_results.extend(benchmark_generator(model_name, test_data))
            except Exception as e:
                print(f"Failed to benchmark {model_name}: {str(e)}")
                continue
        
        save_results(generator_results, GENERATOR_RESULTS)
        
    except Exception as e:
        print(f"Main process error: {str(e)}")
        raise
    finally:
        print("\nBenchmark completed")
        print_memory_usage()

if __name__ == "__main__":
    main()