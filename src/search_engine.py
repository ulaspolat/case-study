# Search engine for hotel room descriptions
import os
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from sentence_transformers import SentenceTransformer

# Try to import SentenceTransformer from the latest version

class SearchEngine:
    def __init__(self, descriptions_dir="descriptions"):
        """Initialize the search engine with description data."""
        # Load descriptions
        self.descriptions_dir = Path(descriptions_dir)
        self.all_descriptions = self._load_descriptions()
        
        # Initialize models for searching
        self.use_semantic_search = True
        try:
            # Using a more powerful model for better semantic understanding
            self.sentence_model = SentenceTransformer('all-mpnet-base-v2')
            print("Using advanced semantic model: all-mpnet-base-v2")
        except Exception as e:
            print(f"Warning: Could not initialize sentence transformer: {e}")
            print("Semantic search will be disabled")
            self.use_semantic_search = False
        
        # Prepare data for searching
        self.prepare_search_data()
    
    def _load_descriptions(self):
        """Load all image descriptions from the JSON files."""
        all_desc_path = self.descriptions_dir / "all_descriptions.json"
        
        if all_desc_path.exists():
            with open(all_desc_path, "r") as f:
                return json.load(f)
        
        # If all_descriptions.json doesn't exist, load individual files
        descriptions = {}
        for json_file in self.descriptions_dir.glob("*.json"):
            if json_file.name == "all_descriptions.json":
                continue
            
            with open(json_file, "r") as f:
                desc_data = json.load(f)
                img_num = json_file.stem
                descriptions[img_num] = desc_data
        
        return descriptions
    
    def prepare_search_data(self):
        """Prepare the search data from the descriptions."""
        self.documents = []
        self.document_ids = []
        self.url_mapping = {}  # Map image IDs to URLs
        
        # Process each description
        for img_id, data in self.all_descriptions.items():
            if "error" in data:
                continue
            
            # Get the full description text
            description = data.get("description", "")
            
            # Store the image URL if available
            if "image_url" in data:
                self.url_mapping[img_id] = data["image_url"]
            
            # Add to documents for keyword search
            self.documents.append(description)
            self.document_ids.append(img_id)
        
        # Create TFIDF vectorizer for keyword search
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
        
        # Create embeddings for semantic search (cache them to avoid recomputing)
        if self.use_semantic_search:
            print("Generating sentence embeddings... (this may take a moment)")
            self.embeddings = self.sentence_model.encode(self.documents, show_progress_bar=True)
            print("Sentence embeddings generated successfully!")
    
    def keyword_search(self, query, top_k=5):
        """
        Perform keyword-based search using TF-IDF.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of image IDs matching the query
        """
        # Convert query to TFIDF vector
        query_vec = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity = (query_vec * self.tfidf_matrix.T).toarray()[0]
        
        # Sort by similarity score
        top_indices = similarity.argsort()[::-1][:top_k]
        
        # Return top matching images
        return [(self.document_ids[i], float(similarity[i])) for i in top_indices if similarity[i] > 0]
    
    def semantic_search(self, query, top_k=5):
        """
        Perform semantic search using sentence embeddings.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of image IDs matching the query
        """
        if not self.use_semantic_search:
            print("Semantic search is disabled. Using keyword search instead.")
            return self.keyword_search(query, top_k)
        
        # Encode the query using our local sentence transformer model
        query_embedding = self.sentence_model.encode(query)
        
        # Calculate cosine similarity
        similarity = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Sort by similarity score
        top_indices = similarity.argsort()[::-1][:top_k]
        
        # Return top matching images
        return [(self.document_ids[i], float(similarity[i])) for i in top_indices if similarity[i] > 0.5]
    
    def search(self, query, structured_data=None, method="combined", top_k=5):
        """
        Combined search method that integrates keyword and semantic approaches.
        
        Args:
            query: Search query (string)
            structured_data: Optional data from query parser (containing original_query and enhanced_query)
            method: Search method ('keyword', 'semantic', or 'combined')
            top_k: Number of top results to return
            
        Returns:
            List of tuples (image_id, score, method_name) for the top matching images
        """
        # If structured_data is provided, use its enhanced_query field when available
        if structured_data is not None:
            if "enhanced_query" in structured_data:
                query_text = structured_data["enhanced_query"]
            elif "original_query" in structured_data:
                query_text = structured_data["original_query"]
            else:
                query_text = query
        else:
            query_text = query
            
        # Get results using keyword and/or semantic search
        results = []
        
        if method in ["keyword", "combined"]:
            keyword_results = self.keyword_search(query_text, top_k=top_k)
            results.extend([(img_id, score, "keyword") for img_id, score in keyword_results])
        
        if method in ["semantic", "combined"] and self.use_semantic_search:
            semantic_results = self.semantic_search(query_text, top_k=top_k)
            results.extend([(img_id, score, "semantic") for img_id, score in semantic_results])
        
        # Deduplicate results, prioritizing higher scores
        unique_results = {}
        for img_id, score, method_name in results:
            if img_id not in unique_results or score > unique_results[img_id][0]:
                unique_results[img_id] = (score, method_name)
        
        # Convert to final output format
        final_results = [(img_id, score, method_name) for img_id, (score, method_name) in unique_results.items()]
        
        # Sort by score, descending
        sorted_results = sorted(final_results, key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return sorted_results[:top_k]
    
    def get_image_urls(self, image_ids):
        """
        Convert image IDs to URLs from the original dataset.
        
        Args:
            image_ids: List of image IDs
            
        Returns:
            List of image URLs
        """
        urls = []
        for img_id in image_ids:
            if img_id in self.url_mapping:
                urls.append(self.url_mapping[img_id])
            else:
                # Fallback to constructing URL
                urls.append(f"https://static.obilet.com.s3.eu-central-1.amazonaws.com/CaseStudy/HotelImages/{img_id}.jpg")
        return urls
        
    def get_descriptions(self, image_ids):
        """
        Get the full description data for a list of image IDs.
        
        Args:
            image_ids: List of image IDs
            
        Returns:
            Dictionary mapping image IDs to their description data
        """
        descriptions = {}
        for img_id in image_ids:
            if img_id in self.all_descriptions:
                descriptions[img_id] = self.all_descriptions[img_id]
        return descriptions
    
    def query_to_urls(self, query, top_k=5):
        """
        Process a query and return matching image URLs.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of image URLs
        """
        results = self.search(query, method="combined", top_k=top_k)
        image_ids = [img_id for img_id, _, _ in results]
        return self.get_image_urls(image_ids)


# Example usage
if __name__ == "__main__":
    engine = SearchEngine()
    
    # Test queries from the case study
    test_queries = [
        "Double rooms with a sea view",
        "Rooms with a balcony and air conditioning, with a city view",
        "Triple rooms with a desk",
        "Rooms with a maximum capacity of 4 people"
    ]
    
    print("Testing search engine with case study queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = engine.query_to_urls(query, top_k=5)
        print(f"Matching images: {results}") 