import numpy as np
import faiss
import json
import os
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any, Optional

class VectorDB:
    """
    Vector database implementation using FAISS for efficient similarity search.
    This class handles storage and retrieval of embeddings for both descriptions and queries.
    """
    def __init__(self, db_path: str = "vector_db"):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to store the vector database files
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize index and metadata storage
        self.index = None
        self.document_ids = []
        self.query_cache = {}
        self.index_path = self.db_path / "faiss_index.bin"
        self.metadata_path = self.db_path / "metadata.pkl"
        self.query_cache_path = self.db_path / "query_cache.json"
        
        # Initialize query vector database
        self.query_index = None
        self.query_texts = []
        self.query_index_path = self.db_path / "query_index.bin"
        self.query_metadata_path = self.db_path / "query_metadata.pkl"
        
        # Load existing data if available
        self.load()
    
    def load(self):
        """Load index and metadata from disk if they exist."""
        try:
            # Load document index
            if self.index_path.exists() and self.metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.document_ids = metadata.get('document_ids', [])
                
                print(f"Loaded vector database with {len(self.document_ids)} documents")
                
                # Load query cache if available
                if self.query_cache_path.exists():
                    with open(self.query_cache_path, 'r') as f:
                        self.query_cache = json.load(f)
                    print(f"Loaded query cache with {len(self.query_cache)} entries")
            else:
                print("No existing document vector database found")
            
            # Load query index
            if self.query_index_path.exists() and self.query_metadata_path.exists():
                # Load FAISS query index
                self.query_index = faiss.read_index(str(self.query_index_path))
                
                # Load query metadata
                with open(self.query_metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.query_texts = metadata.get('query_texts', [])
                
                print(f"Loaded query vector database with {len(self.query_texts)} queries")
            else:
                print("No existing query vector database found")
                
        except Exception as e:
            print(f"Error loading vector database: {e}")
            # Initialize empty data structures
            self.index = None
            self.document_ids = []
            self.query_cache = {}
            self.query_index = None
            self.query_texts = []
    
    def save(self):
        """Save index and metadata to disk."""
        try:
            # Save document index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
                
                # Save metadata
                with open(self.metadata_path, 'wb') as f:
                    metadata = {
                        'document_ids': self.document_ids
                    }
                    pickle.dump(metadata, f)
                
                # Save query cache
                with open(self.query_cache_path, 'w') as f:
                    json.dump(self.query_cache, f)
                
                print(f"Saved vector database with {len(self.document_ids)} documents")
            
            # Save query index
            if self.query_index is not None:
                faiss.write_index(self.query_index, str(self.query_index_path))
                
                # Save query metadata
                with open(self.query_metadata_path, 'wb') as f:
                    metadata = {
                        'query_texts': self.query_texts
                    }
                    pickle.dump(metadata, f)
                
                print(f"Saved query vector database with {len(self.query_texts)} queries")
                
        except Exception as e:
            print(f"Error saving vector database: {e}")
    
    def build_index(self, embeddings: np.ndarray, document_ids: List[str]):
        """
        Build a FAISS index from embeddings.
        
        Args:
            embeddings: Array of embedding vectors
            document_ids: List of document IDs corresponding to the embeddings
        """
        # Store document IDs
        self.document_ids = document_ids
        
        # Get dimensionality of embeddings
        dimension = embeddings.shape[1]
        
        # Build FAISS index
        # Using IndexFlatIP for cosine similarity (Inner Product)
        # Convert vectors to unit length for cosine similarity
        normalized_embeddings = self._normalize_vectors(embeddings)
        
        # Create a new index
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(normalized_embeddings)
        
        # Save to disk
        self.save()
    
    def add_query_to_index(self, query_text: str, query_embedding: np.ndarray):
        """
        Add a query to the query vector database.
        
        Args:
            query_text: Original query text
            query_embedding: Query embedding vector
        """
        # Skip if query already exists
        if query_text in self.query_texts:
            return
            
        # Initialize query index if it doesn't exist
        if self.query_index is None:
            dimension = query_embedding.shape[0]
            self.query_index = faiss.IndexFlatIP(dimension)
        
        # Normalize query embedding for cosine similarity
        normalized_embedding = self._normalize_vectors(query_embedding.reshape(1, -1))
        
        # Add to index
        self.query_index.add(normalized_embedding)
        self.query_texts.append(query_text)
        
        # Save after each new query to ensure persistence
        self.save()
        print(f"Added query to vector database: '{query_text}'")
    
    def find_similar_queries(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find queries similar to the given query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (query_text, similarity_score)
        """
        if self.query_index is None or len(self.query_texts) == 0:
            return []
        
        # Limit top_k to the number of queries we have
        top_k = min(top_k, len(self.query_texts))
        
        # Normalize query vector for cosine similarity
        query_embedding = self._normalize_vectors(query_embedding.reshape(1, -1))
        
        # Search the index
        scores, indices = self.query_index.search(query_embedding, top_k)
        
        # Map indices to query texts and scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.query_texts):  # Check if index is valid
                results.append((self.query_texts[idx], float(score)))
        
        return results
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar vectors in the index.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_id, similarity_score)
        """
        if self.index is None:
            return []
        
        # Normalize query vector for cosine similarity
        query_embedding = self._normalize_vectors(query_embedding.reshape(1, -1))
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Map indices to document IDs and scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(self.document_ids):  # Check if index is valid
                results.append((self.document_ids[idx], float(score)))
        
        return results
    
    def cache_query(self, query: str, results: List[Tuple[str, float]]):
        """
        Cache query results to avoid recomputing for identical queries.
        
        Args:
            query: Original query string
            results: Search results to cache
        """
        # Store only the top results
        self.query_cache[query] = results
        
        # Save to disk periodically
        # In a production environment, you might want to implement a more efficient
        # cache saving strategy (e.g., save after N new queries or on application exit)
        if len(self.query_cache) % 10 == 0:
            with open(self.query_cache_path, 'w') as f:
                json.dump(self.query_cache, f)
    
    def get_cached_results(self, query: str) -> Optional[List[Tuple[str, float]]]:
        """
        Retrieve cached results for a query.
        
        Args:
            query: Query string
            
        Returns:
            Cached results if available, None otherwise
        """
        return self.query_cache.get(query)
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the query database.
        
        Returns:
            Dictionary with query statistics
        """
        return {
            "total_queries": len(self.query_texts),
            "unique_queries": len(set(self.query_texts)),
            "cached_queries": len(self.query_cache),
            "has_query_index": self.query_index is not None
        }
    
    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.
        
        Args:
            vectors: Input vectors
            
        Returns:
            Normalized vectors
        """
        # Calculate L2 norm along each row
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-15)
        # Normalize
        return vectors / norms 