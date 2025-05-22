from typing import List, Dict, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class Generator:
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model loaded on device: {self.device}")
    
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with output scores
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                output_scores=True,
                return_dict_in_generate=True,
                **kwargs
            )
        
        # Calculate confidence score (average token probability)
        scores = torch.stack(outputs.scores, dim=1)
        confidence = torch.mean(torch.softmax(scores, dim=-1).max(dim=-1).values).item()
        
        answer = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        # Fallback if confidence is low or answer is generic
        if confidence < 0.3 or "cannot find" in answer.lower():
            answer = "I cannot confidently answer based on the provided context."
        
        return {
            'answer': answer,
            'confidence': round(confidence, 3),
            'is_fallback': confidence < 0.3
        }
    
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