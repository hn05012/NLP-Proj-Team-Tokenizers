from rouge import Rouge 
import numpy as np
import json

class Evaluator:
    def __init__(self):
        self.rouge = Rouge()
    
    def calculate_rouge(self, generated: str, reference: str) -> dict:
        # Handle empty strings or mismatched types
        if not generated.strip() or not reference.strip():
            return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
        try:
            return self.rouge.get_scores(generated, reference)[0]
        except:
            return {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}}
    
    def evaluate_test_set(self, pipeline, test_file: str):
        with open(test_file) as f:
            test_data = json.load(f)
        
        results = []
        for item in test_data:
            try:
                output = pipeline.query(item['question'])
                generated = output['generated_answer'].strip()
                reference = item['expected_answer'].strip()
                
                rouge_scores = self.calculate_rouge(generated, reference)
                
                results.append({
                    'question': item['question'],
                    'generated': generated,
                    'expected': reference,
                    'rouge': rouge_scores,
                    'grounded': output.get('is_grounded', False)  
                })
            except Exception as e:
                print(f"Error processing question '{item['question']}': {str(e)}")
                results.append({
                    'question': item['question'],
                    'error': str(e),
                    'rouge': {'rouge-1': {'f': 0}, 'rouge-2': {'f': 0}, 'rouge-l': {'f': 0}},
                    'grounded': False
                })
        
        # Aggregate metrics
        avg_rouge = {
            'rouge-1': np.mean([r['rouge']['rouge-1']['f'] for r in results if 'rouge' in r]),
            'rouge-2': np.mean([r['rouge']['rouge-2']['f'] for r in results if 'rouge' in r]),
            'rouge-l': np.mean([r['rouge']['rouge-l']['f'] for r in results if 'rouge' in r])
        }
        
        return {
            'per_question': results,
            'average_scores': avg_rouge,
            'success_rate': sum(r['grounded'] for r in results if 'grounded' in r) / len(results)
        }

if __name__ == "__main__":
    from pipeline import RAGPipeline
    pipeline = RAGPipeline()

    sample_texts = [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "Machine learning is a subset of AI that enables systems to learn from data."
    ]
    pipeline.add_documents(sample_texts)
    
    evaluator = Evaluator()
    results = evaluator.evaluate_test_set(pipeline, "data/test_inputs.json")
    
    print("\n📊 Evaluation Results:")
    print(f"ROUGE-1 F1: {results['average_scores']['rouge-1']:.3f}")
    print(f"ROUGE-2 F1: {results['average_scores']['rouge-2']:.3f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    
    # Print failures
    failures = [r for r in results['per_question'] if r['rouge']['rouge-1']['f'] < 0.5]
    if failures:
        print("\n🔴 Problematic Questions:")
        for f in failures[:3]:  # Show top 3 worst
            print(f"Q: {f['question']}")
            print(f"Generated: {f.get('generated', '')}")
            print(f"Expected: {f.get('expected', '')}")
            print(f"ROUGE-1: {f['rouge']['rouge-1']['f']:.3f}\n")