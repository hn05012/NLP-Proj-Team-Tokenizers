import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any
import logging
from retriever.retriever import Retriever
from generator.generator import Generator

class RAGPipeline:
    def __init__(self, retriever_model: str = 'all-MiniLM-L6-v2', 
                 generator_model: str = 'google/flan-t5-small',
                 log_file: str = 'logs/rag_logs.jsonl'):
        self.retriever = Retriever(model_name=retriever_model)
        self.generator = Generator(model_name=generator_model)
        self.log_file = log_file
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.logger = logging.getLogger(__name__)
        self.group_id = str(uuid.uuid4())[:8]
        
        print(f"RAG Pipeline initialized with group_id: {self.group_id}")
    
    def add_documents(self, documents: List[str]) -> None:

        self.logger.info(f"Adding {len(documents)} documents to retriever")
        self.retriever.add_documents(documents)
        self.logger.info("Documents successfully added and indexed")
    
    def query(self, question: str, k: int = 5, **generation_kwargs) -> Dict[str, Any]:

        timestamp = datetime.now().isoformat()
        self.logger.info(f"Retrieving chunks for question: {question[:100]}...")
        retrieved_chunks = self.retriever.query(question, k=k)
        
        self.logger.info("Generating answer...")
        generation_result = self.generator.answer_question(question, retrieved_chunks, **generation_kwargs)
        result = {
            'question': question,
            'retrieved_chunks': retrieved_chunks,
            'prompt': generation_result['prompt'],
            'generated_answer': generation_result['answer'],
            'timestamp': timestamp,
            'group_id': self.group_id,
            'num_chunks_used': len(retrieved_chunks),
            'context_sources': generation_result['context_sources']
        }
        self._log_query(result)
        
        return result
    
    def _log_query(self, result: Dict[str, Any]) -> None:

        log_entry = {
            'question': result['question'],
            'retrieved_chunks': [
                {
                    'chunk': chunk['chunk'][:200] + '...' if len(chunk['chunk']) > 200 else chunk['chunk'],
                    'score': chunk['score'],
                    'source': chunk['source']
                }
                for chunk in result['retrieved_chunks']
            ],
            'prompt': result['prompt'][:500] + '...' if len(result['prompt']) > 500 else result['prompt'],
            'generated_answer': result['generated_answer'],
            'timestamp': result['timestamp'],
            'group_id': result['group_id']
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def save_retriever(self, filepath: str) -> None:
        self.retriever.save(filepath)
    
    def load_retriever(self, filepath: str) -> None:
        self.retriever.load(filepath)
    
    def run_test_questions(self, test_file: str = 'data/test_inputs.json') -> Dict[str, Any]:
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = []
        successful_queries = 0
        
        print(f"Running {len(test_data)} test questions...")
        
        for i, test_item in enumerate(test_data, 1):
            question = test_item['question']
            expected_answer = test_item.get('expected_answer', '')
            
            print(f"\nTest {i}/{len(test_data)}: {question}")
            
            try:
                result = self.query(question)
                answer = result['generated_answer']
                
                # Basic checks
                has_answer = len(answer.strip()) > 0
                not_default_response = "cannot find" not in answer.lower()
                grounded_in_context = self._check_grounding(answer, result['retrieved_chunks'])
                
                if has_answer and (not_default_response or grounded_in_context):
                    successful_queries += 1
                
                results.append({
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_answer': answer,
                    'has_answer': has_answer,
                    'grounded_in_context': grounded_in_context,
                    'success': has_answer and (not_default_response or grounded_in_context)
                })
                
                print(f"Answer: {answer}")
                print(f"Success: {has_answer and (not_default_response or grounded_in_context)}")
                
            except Exception as e:
                print(f"Error processing question: {e}")
                results.append({
                    'question': question,
                    'expected_answer': expected_answer,
                    'generated_answer': '',
                    'error': str(e),
                    'success': False
                })
        
        test_results = {
            'total_questions': len(test_data),
            'successful_queries': successful_queries,
            'success_rate': successful_queries / len(test_data) if test_data else 0,
            'individual_results': results
        }
        
        results_file = f'logs/test_results_{self.group_id}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nTest Results Summary:")
        print(f"Total questions: {test_results['total_questions']}")
        print(f"Successful queries: {test_results['successful_queries']}")
        print(f"Success rate: {test_results['success_rate']:.2%}")
        print(f"Results saved to: {results_file}")
        
        return test_results
    
    def _check_grounding(self, answer: str, chunks: List[Dict[str, Any]]) -> bool:
        if not answer or len(answer.strip()) < 5:
            return False
        context_text = ' '.join([chunk['chunk'].lower() for chunk in chunks])
        answer_words = answer.lower().split()
        
        matching_words = sum(1 for word in answer_words if word in context_text and len(word) > 3)
        
        return matching_words / len(answer_words) > 0.3 if answer_words else False

def demo_pipeline():
    print("=== Team Tokenizers RAG Pipeline Demo ===")

    pipeline = RAGPipeline()
    
    sample_docs = [
        """
        Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. 
        It involves algorithms that can identify patterns, make predictions, and improve their performance over time through experience.
        
        There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
        Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data.
        Reinforcement learning involves agents learning through interaction with an environment.
        """,
        """
        Python is a high-level programming language known for its simplicity and readability. 
        It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, 
        including procedural, object-oriented, and functional programming.
        
        Python is widely used in web development, data science, artificial intelligence, automation, and scientific computing.
        Popular libraries include NumPy for numerical computing, Pandas for data manipulation, and TensorFlow for machine learning.
        """
    ]
    
    print("Adding sample documents...")
    pipeline.add_documents(sample_docs)
    
    questions = [
        "What is machine learning?",
        "Who created Python?",
        "What are the types of machine learning?",
        "What is Python used for?"
    ]
    
    for question in questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print('='*50)
        
        result = pipeline.query(question)
        
        print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
        for i, chunk in enumerate(result['retrieved_chunks'][:2], 1):
            print(f"  Chunk {i} (score: {chunk['score']:.3f}): {chunk['chunk'][:100]}...")
        
        print(f"\nGenerated Answer: {result['generated_answer']}")

if __name__ == "__main__":
    demo_pipeline()