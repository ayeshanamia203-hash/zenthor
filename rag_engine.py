# ============================================================
# RAG ENGINE FOR ZENTHOR
# Document Retrieval + AI Answer Generation
# ============================================================

import os
import uuid
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# ============================================================
# EMBEDDING MODEL
# ============================================================

# Using a lightweight, free embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class RAGEngine:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB for persistent storage
        """
        self.persist_directory = persist_directory
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="zenthor_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
    
    def add_document(self, text: str, metadata: dict = None):
        """
        Add a document to the vector database
        """
        if not text or len(text.strip()) < 10:
            return None
        
        # Generate unique ID
        doc_id = str(uuid.uuid4())
        
        # Split text into chunks (max 500 chars each)
        chunks = self._chunk_text(text, chunk_size=500)
        
        for i, chunk in enumerate(chunks):
            self.collection.add(
                documents=[chunk],
                metadatas=[metadata or {}],
                ids=[f"{doc_id}_{i}"]
            )
        
        return doc_id
    
    def _chunk_text(self, text: str, chunk_size: int = 500):
        """
        Split text into overlapping chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search for relevant documents
        """
        if not query or not query.strip():
            return []
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Extract documents from results
        documents = results.get('documents', [[]])[0]
        return documents if documents else []
    
    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        Get context for RAG
        """
        docs = self.search(query, top_k)
        if not docs:
            return ""
        
        # Combine documents with separators
        context = "\n\n---\n\n".join(docs)
        return context
    
    def clear_all(self):
        """
        Delete all documents
        """
        self.client.delete_collection("zenthor_docs")
        self.collection = self.client.get_or_create_collection(
            name="zenthor_docs",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
