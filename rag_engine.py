# ============================================================
# RAG ENGINE - LIGHTWEIGHT FALLBACK FOR RENDER
# No chromadb / sentence-transformers required
# Uses simple keyword matching for document retrieval
# ============================================================

import uuid
from typing import List, Optional


class RAGEngine:
    """
    A lightweight in-memory RAG engine that stores documents
    and retrieves them using simple keyword matching.
    Perfect for Render's free tier (low memory usage).
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the RAG engine.
        The persist_directory parameter is kept for compatibility
        but not used (in-memory only).
        """
        self._documents = []
        print("RAG Engine initialized (lightweight fallback mode).")

    def add_document(self, text: str, metadata: dict = None) -> Optional[str]:
        """
        Add a document to the in-memory store.

        Args:
            text: The document content.
            metadata: Optional metadata (filename, user, etc.)

        Returns:
            A unique document ID, or None if text is invalid.
        """
        if not text or len(text.strip()) < 10:
            return None

        doc_id = str(uuid.uuid4())
        self._documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata or {}
        })
        print(f"RAG: Document added successfully (ID: {doc_id[:8]})")
        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Search for documents relevant to the query.

        Uses simple keyword frequency matching.
        For a production system, you would replace this with
        proper vector search (e.g., ChromaDB, Pinecone).

        Args:
            query: The search query.
            top_k: Number of top results to return.

        Returns:
            List of document texts.
        """
        if not query or not self._documents:
            return []

        # Simple keyword matching
        query_words = set(query.lower().split())
        results = []

        for doc in self._documents:
            text_lower = doc["text"].lower()
            # Count how many query words appear in the document
            score = sum(1 for word in query_words if word in text_lower)
            results.append((score, doc["text"]))

        # Sort by score (highest first)
        results.sort(key=lambda x: x[0], reverse=True)

        # Return top_k results
        return [text for _, text in results[:top_k]]

    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        Get combined context from relevant documents.

        Args:
            query: The search query.
            top_k: Number of top results to combine.

        Returns:
            Combined context string, or empty string if no results.
        """
        docs = self.search(query, top_k)
        if not docs:
            return ""

        # Combine documents with a separator
        return "\n\n---\n\n".join(docs)

    def clear_all(self):
        """
        Delete all documents from memory.
        """
        self._documents = []
        print("RAG: All documents cleared.")

    def get_document_count(self) -> int:
        """
        Get the number of stored documents.
        """
        return len(self._documents)
