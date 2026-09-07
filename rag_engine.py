# ============================================================
# RAG ENGINE FOR ZENTHOR (In-Memory Mode for Render)
# ============================================================

import uuid
from typing import List, Optional

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("chromadb not installed — RAG will use simple fallback.")

class RAGEngine:
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize RAG with optional persistence.
        If persist_directory is None, uses in-memory.
        """
        self.persist_directory = persist_directory
        self.collection = None
        self._init_collection()

    def _init_collection(self):
        """Initialize ChromaDB collection (in-memory or persistent)."""
        if not CHROMADB_AVAILABLE:
            # Simple fallback: store documents in memory as a list
            self._documents = []
            self._fallback_mode = True
            return

        try:
            # Use in-memory client if persist_directory is None
            if self.persist_directory is None:
                self.client = chromadb.Client()
            else:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory
                )

            # Use SentenceTransformer embedding
            embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="zenthor_docs",
                embedding_function=embedding_fn
            )
            self._fallback_mode = False
            print("ChromaDB initialized successfully.")

        except Exception as e:
            print(f"ChromaDB init failed: {str(e)} — using fallback.")
            self._documents = []
            self._fallback_mode = True

    def add_document(self, text: str, metadata: dict = None):
        """Add document to the vector store."""
        if not text or len(text.strip()) < 10:
            return None

        # Fallback mode: just store in list
        if self._fallback_mode:
            doc_id = str(uuid.uuid4())
            self._documents.append({
                "id": doc_id,
                "text": text,
                "metadata": metadata or {}
            })
            return doc_id

        # ChromaDB mode
        try:
            doc_id = str(uuid.uuid4())
            chunks = self._chunk_text(text, chunk_size=500)

            for i, chunk in enumerate(chunks):
                self.collection.add(
                    documents=[chunk],
                    metadatas=[metadata or {}],
                    ids=[f"{doc_id}_{i}"]
                )

            return doc_id

        except Exception as e:
            print(f"RAG add_document error: {str(e)}")
            return None

    def _chunk_text(self, text: str, chunk_size: int = 500):
        """Split text into overlapping chunks."""
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
        """Search for relevant documents."""
        if not query or not query.strip():
            return []

        # Fallback mode: simple keyword search
        if self._fallback_mode:
            if not self._documents:
                return []

            # Simple keyword matching
            query_words = set(query.lower().split())
            results = []

            for doc in self._documents:
                text_lower = doc["text"].lower()
                score = sum(1 for word in query_words if word in text_lower)
                results.append((score, doc["text"]))

            results.sort(key=lambda x: x[0], reverse=True)
            return [text for score, text in results[:top_k]]

        # ChromaDB mode
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            documents = results.get('documents', [[]])[0]
            return documents if documents else []

        except Exception as e:
            print(f"RAG search error: {str(e)}")
            return []

    def get_context(self, query: str, top_k: int = 5) -> str:
        """Get combined context from relevant documents."""
        docs = self.search(query, top_k)
        if not docs:
            return ""

        return "\n\n---\n\n".join(docs)

    def clear_all(self):
        """Delete all documents."""
        if self._fallback_mode:
            self._documents = []
            return

        try:
            self.client.delete_collection("zenthor_docs")
            self.collection = self.client.get_or_create_collection(
                name="zenthor_docs",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
        except Exception as e:
            print(f"RAG clear error: {str(e)}")
