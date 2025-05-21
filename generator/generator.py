from typing import List, Dict, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class Generator:
    def __init__(self, model_name: str = 'google/flan-t5-base'):
        """
        Initialize the Generator with a T5 model.
        
        Args:
            model_name: Name of the T5 model to use
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Use CPU or GPU based on availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model loaded on device: {self.device}")
    
    def build_prompt(self, question: str, context_chunks: List[Dict[str, Any]], max_context_length: int = 1000) -> str:
        """
        Build a prompt from the question and retrieved context chunks.
        
        Args:
            question: The input question
            context_chunks: List of retrieved chunks with metadata
            max_context_length: Maximum length of context to include
            
        Returns:
            Formatted prompt string
        """
        # Sort chunks by relevance score (descending)
        sorted_chunks = sorted(context_chunks, key=lambda x: x['score'], reverse=True)
        
        # Build context from relevant chunks
        context_parts = []
        current_length = 0
        
        for chunk_info in sorted_chunks:
            chunk_text = chunk_info['chunk']
            # Add some context about the source
            chunk_with_source = f"[Source: {chunk_info['source']}] {chunk_text}"
            
            if current_length + len(chunk_with_source) <= max_context_length:
                context_parts.append(chunk_with_source)
                current_length += len(chunk_with_source)
            else:
                # Add partial chunk if there's room
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if there's meaningful space
                    context_parts.append(chunk_with_source[:remaining_space] + "...")
                break
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt
        prompt = f"""Answer the following question based on the provided context. If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str, max_length: int = 200, temperature: float = 0.7, 
                       num_beams: int = 4, do_sample: bool = True) -> str:
        """
        Generate an answer from the formatted prompt.
        
        Args:
            prompt: The formatted prompt containing question and context
            max_length: Maximum length of the generated answer
            temperature: Sampling temperature (higher = more random)
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            
        Returns:
            Generated answer string
        """
        # Tokenize the prompt
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=1024, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate answer
        with torch.no_grad():
            if do_sample:
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode the generated text
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the answer (remove any repeated prompt text)
        answer = answer.strip()
        
        return answer
    
    def answer_question(self, question: str, context_chunks: List[Dict[str, Any]], 
                       **generation_kwargs) -> Dict[str, Any]:
        """
        Complete pipeline: build prompt and generate answer.
        
        Args:
            question: The input question
            context_chunks: Retrieved context chunks
            **generation_kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing prompt, answer, and metadata
        """
        # Build the prompt
        prompt = self.build_prompt(question, context_chunks)
        
        # Generate the answer
        answer = self.generate_answer(prompt, **generation_kwargs)
        
        return {
            'question': question,
            'prompt': prompt,
            'answer': answer,
            'num_context_chunks': len(context_chunks),
            'context_sources': [chunk['source'] for chunk in context_chunks]
        }