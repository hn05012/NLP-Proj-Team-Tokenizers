import os
import json
import pickle
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from pathlib import Path

class Retriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the Retriever with sentence transformer model and chunking parameters.
        Uses sklearn instead of FAISS for compatibility.
        
        Args:
            model_name: Name of the SentenceTransformer model
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents = []
        self.chunks = []
        self.embeddings = None
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentences first to maintain semantic coherence
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _load_text_file(self, file_path: str) -> str:
        """Load text from .txt or .md file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_pdf_file(self, file_path: str) -> str:
        """Load text from .pdf file."""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the retriever. Documents can be file paths or raw text.
        
        Args:
            documents: List of file paths or text strings
        """
        for doc in documents:
            # Check if it's a file path
            if os.path.exists(doc):
                file_ext = Path(doc).suffix.lower()
                if file_ext == '.pdf':
                    text = self._load_pdf_file(doc)
                elif file_ext in ['.txt', '.md']:
                    text = self._load_text_file(doc)
                else:
                    raise ValueError(f"Unsupported file type: {file_ext}")
                
                # Store document info
                doc_info = {
                    'source': doc,
                    'type': 'file',
                    'content': text
                }
                self.documents.append(doc_info)
            else:
                # Treat as raw text
                doc_info = {
                    'source': 'raw_text',
                    'type': 'text',
                    'content': doc
                }
                self.documents.append(doc_info)
        
        # Chunk all documents and create embeddings
        self._create_chunks_and_embeddings()
    
    def _create_chunks_and_embeddings(self) -> None:
        """Create chunks from all documents and generate embeddings."""
        all_chunks = []
        chunk_metadata = []
        
        for doc_idx, doc in enumerate(self.documents):
            chunks = self._chunk_text(doc['content'])
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'source': doc['source']
                })
        
        self.chunks = all_chunks
        self.chunk_metadata = chunk_metadata
        
        # Generate embeddings
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        self.embeddings = self.model.encode(all_chunks)
        
        print(f"Created embedding matrix with shape: {self.embeddings.shape}")
    
    def query(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a question using cosine similarity.
        
        Args:
            question: Query string
            k: Number of chunks to retrieve
            
        Returns:
            List of dictionaries containing chunk info and scores
        """
        if self.embeddings is None:
            raise ValueError("No documents have been added yet")
        
        # Encode query
        query_embedding = self.model.encode([question])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'score': float(similarities[idx]),
                'source': self.chunk_metadata[idx]['source'],
                'doc_idx': self.chunk_metadata[idx]['doc_idx'],
                'chunk_idx': self.chunk_metadata[idx]['chunk_idx']
            })
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save the retriever state to disk.
        
        Args:
            filepath: Path to save the retriever
        """
        state = {
            'documents': self.documents,
            'chunks': self.chunks,
            'chunk_metadata': self.chunk_metadata,
            'embeddings': self.embeddings,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Retriever saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load the retriever state from disk.
        
        Args:
            filepath: Path to load the retriever from
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.documents = state['documents']
        self.chunks = state['chunks']
        self.chunk_metadata = state['chunk_metadata']
        self.embeddings = state['embeddings']
        self.chunk_size = state['chunk_size']
        self.chunk_overlap = state['chunk_overlap']
        
        # Reinitialize model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Retriever loaded from {filepath}")