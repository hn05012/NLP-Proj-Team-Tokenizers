import csv
import time
import psutil
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Config
MODELS = {
    "MiniLM": "all-MiniLM-L6-v2",
    "BGE-small": "BAAI/bge-small-en-v1.5",
    "E5-base": "intfloat/e5-base-v2"
}
TEST_FILE = "data/test_inputs.json"
OUTPUT_CSV = "results/retriever_results.csv"

def load_test_data(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    # Add mock context chunks if not present
    for item in data:
        if "context_chunks" not in item:
            item["context_chunks"] = [
                {"text": item["expected_answer"], "contains_answer": True},
                {"text": "Python is a programming language.", "contains_answer": False}
            ]
    return data

def benchmark_model(model_name, test_data):
    print(f"Benchmarking {model_name}...")
    model = SentenceTransformer(model_name)
    
    results = []
    for query in test_data:
        # Prepare documents
        docs = [chunk["text"] for chunk in query["context_chunks"]]
        gold_chunks = [i for i, chunk in enumerate(query["context_chunks"]) 
                      if chunk["contains_answer"]]
        
        # Encode and compute similarities
        start_time = time.perf_counter()
        doc_embeddings = model.encode(docs)
        query_embedding = model.encode([query["question"]])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        latency = (time.perf_counter() - start_time) * 1000  # ms
        
        # Calculate metrics
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

if __name__ == "__main__":
    test_data = load_test_data(TEST_FILE)
    all_results = []
    
    for model_name in MODELS.values():
        all_results.extend(benchmark_model(model_name, test_data))
    
    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"Results saved to {OUTPUT_CSV}")