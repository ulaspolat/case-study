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
        Can also use structured data from the query parser for better results.
        
        Args:
            query: Search query (string)
            structured_data: Optional structured data from query parser
            method: Search method ('keyword', 'semantic', or 'combined')
            top_k: Number of top results to return
            
        Returns:
            List of image URLs matching the query
        """
        # If structured_data is None, initialize it with just the original query
        if structured_data is None:
            structured_data = {"original_query": query}
            
        # Get the original query text
        query_text = structured_data.get("original_query", query)
        
        # Get initial results using keyword and/or semantic search
        results = []
        
        if method in ["keyword", "combined"]:
            keyword_results = self.keyword_search(query_text, top_k=top_k*2)  # Get more results for better filtering
            results.extend([(img_id, score, "keyword") for img_id, score in keyword_results])
        
        if method in ["semantic", "combined"] and self.use_semantic_search:
            semantic_results = self.semantic_search(query_text, top_k=top_k*2)  # Get more results for better filtering
            results.extend([(img_id, score, "semantic") for img_id, score in semantic_results])
        
        # Deduplicate results, prioritizing higher scores
        unique_results = {}
        for img_id, score, method_name in results:
            if img_id not in unique_results or score > unique_results[img_id][0]:
                unique_results[img_id] = (score, method_name)
        
        # Get all unique image IDs
        image_ids = list(unique_results.keys())
        
        # Get descriptions for feature-based scoring
        descriptions = self.get_descriptions(image_ids)
        
        # Extract key features from the query and calculate feature scores
        feature_scores = self._calculate_feature_scores(query_text, descriptions)
        
        # If we have structured data, use it to enhance the feature scores
        if len(structured_data) > 1:  # More than just the original query
            structured_scores = self._calculate_structured_scores(structured_data, descriptions)
            
            # Combine feature scores with structured scores (giving more weight to structured scores)
            for img_id in feature_scores:
                if img_id in structured_scores:
                    feature_scores[img_id] = feature_scores[img_id] * 0.4 + structured_scores[img_id] * 0.6
        
        # Combine base scores with feature scores
        final_scores = []
        for img_id, (base_score, method_name) in unique_results.items():
            # Combine scores: 70% base score + 30% feature score
            if img_id in feature_scores:
                combined_score = base_score * 0.7 + feature_scores[img_id] * 0.3
            else:
                combined_score = base_score
                
            final_scores.append((img_id, combined_score, method_name))
        
        # Sort by combined score, descending
        sorted_results = sorted(final_scores, key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return sorted_results[:top_k]
        
    def _calculate_feature_scores(self, query, descriptions):
        """
        Calculate feature-based scores for each room based on query matching.
        
        Args:
            query: Search query
            descriptions: Dictionary of room descriptions
            
        Returns:
            Dictionary mapping image IDs to feature scores
        """
        feature_scores = {}
        
        # Define key features to look for
        key_features = [
            "double", "single", "triple", "suite",  # Room types
            "sea view", "city view", "garden view", "mountain view",  # View types
            "balcony", "air conditioning", "desk", "tv", "minibar"  # Common features
        ]
        
        # Lowercase query for case-insensitive matching
        query_lower = query.lower()
        
        # Check for "and" constructs in the query
        and_features = []
        for i in range(len(key_features)):
            for j in range(i+1, len(key_features)):
                feature1, feature2 = key_features[i], key_features[j]
                if f"{feature1} and {feature2}" in query_lower or f"{feature2} and {feature1}" in query_lower:
                    and_features.append((feature1, feature2))
        
        # Calculate scores for each room
        for img_id, desc_data in descriptions.items():
            score = 0.0
            matches = 0
            total_checks = 0
            
            # Get the description text
            desc_text = desc_data.get("description", "").lower()
            
            # Check for individual features
            for feature in key_features:
                if feature in query_lower:  # Only check features mentioned in the query
                    total_checks += 1
                    if feature in desc_text:
                        matches += 1
                        score += 1.0
            
            # Check for "and" constructs (both features must be present)
            for feature1, feature2 in and_features:
                total_checks += 1  # Count as one additional check
                if feature1 in desc_text and feature2 in desc_text:
                    matches += 1
                    score += 2.0  # Give extra weight to matching "and" constructs
            
            # Normalize score
            if total_checks > 0:
                feature_scores[img_id] = score / total_checks
            else:
                feature_scores[img_id] = 0.5  # Neutral score if no features checked
        
        return feature_scores
        
    def _calculate_structured_scores(self, structured_data, descriptions):
        """
        Calculate scores based on structured data from the query parser.
        This provides more precise matching than text-based feature matching.
        
        Args:
            structured_data: Structured query data from the query parser
            descriptions: Dictionary of room descriptions
            
        Returns:
            Dictionary mapping image IDs to structured match scores
        """
        structured_scores = {}
        
        for img_id, desc_data in descriptions.items():
            score = 0.0
            fields_checked = 0
            
            # Check room type
            if structured_data.get('room_type') and desc_data.get('room_type'):
                fields_checked += 1
                if structured_data['room_type'].lower() in desc_data['room_type'].lower():
                    score += 1.0
            
            # Check min capacity
            if structured_data.get('min_capacity') and desc_data.get('max_capacity'):
                fields_checked += 1
                query_min_capacity = structured_data['min_capacity']
                desc_capacity = desc_data['max_capacity']
                
                # Try to convert to integers for comparison
                try:
                    if isinstance(query_min_capacity, str):
                        query_min_capacity = int(''.join(filter(str.isdigit, query_min_capacity)))
                    if isinstance(desc_capacity, str):
                        desc_capacity = int(''.join(filter(str.isdigit, desc_capacity)))
                    
                    # Room must accommodate at least the minimum number of people
                    if query_min_capacity <= desc_capacity:
                        score += 1.0
                except (ValueError, TypeError):
                    # If conversion fails, do a string comparison
                    if str(query_min_capacity) == str(desc_capacity):
                        score += 1.0
            
            # Check max capacity
            if structured_data.get('max_capacity') and desc_data.get('max_capacity'):
                fields_checked += 1
                query_max_capacity = structured_data['max_capacity']
                desc_capacity = desc_data['max_capacity']
                
                # Try to convert to integers for comparison
                try:
                    if isinstance(query_max_capacity, str):
                        query_max_capacity = int(''.join(filter(str.isdigit, query_max_capacity)))
                    if isinstance(desc_capacity, str):
                        desc_capacity = int(''.join(filter(str.isdigit, desc_capacity)))
                    
                    # Room should not exceed the maximum capacity
                    if desc_capacity <= query_max_capacity:
                        score += 1.0
                except (ValueError, TypeError):
                    # If conversion fails, do a string comparison
                    if str(query_max_capacity) == str(desc_capacity):
                        score += 1.0
            
            # Check view type
            if structured_data.get('view_type') and desc_data.get('view_type'):
                fields_checked += 1
                if structured_data['view_type'].lower() in desc_data['view_type'].lower():
                    score += 1.0
            
            # Check features (all must be present)
            if structured_data.get('features') and desc_data.get('features'):
                query_features = structured_data['features']
                desc_features = desc_data.get('features', [])
                desc_text = desc_data.get('description', '').lower()
                
                # Convert to list if needed
                if isinstance(query_features, str):
                    query_features = [query_features]
                if isinstance(desc_features, str):
                    desc_features = [desc_features]
                
                # For 'and' logic, all features must be present
                fields_checked += 1
                all_features_found = True
                
                for feature in query_features:
                    feature_found = False
                    # Check in features list
                    for desc_feature in desc_features:
                        if feature.lower() in desc_feature.lower():
                            feature_found = True
                            break
                    
                    # If not found in features list, check in description text
                    if not feature_found and feature.lower() in desc_text:
                        feature_found = True
                    
                    # If any feature is missing, the 'and' condition fails
                    if not feature_found:
                        all_features_found = False
                        break
                
                # Only give a score if ALL features are found
                if all_features_found:
                    score += 1.0
            
            # Check other fields (room_size, bed_configuration, design_style, extra_amenities)
            for field in ['room_size', 'bed_configuration', 'design_style']:
                if structured_data.get(field) and desc_data.get(field):
                    fields_checked += 1
                    if structured_data[field].lower() in desc_data[field].lower():
                        score += 1.0
            
            # Check extra amenities
            if structured_data.get('extra_amenities') and desc_data.get('extra_amenities'):
                query_amenities = structured_data['extra_amenities']
                desc_amenities = desc_data.get('extra_amenities', [])
                desc_text = desc_data.get('description', '').lower()
                
                # Convert to list if needed
                if isinstance(query_amenities, str):
                    query_amenities = [query_amenities]
                if isinstance(desc_amenities, str):
                    desc_amenities = [desc_amenities]
                
                # Check each amenity
                for amenity in query_amenities:
                    fields_checked += 1
                    amenity_found = False
                    
                    # Check in amenities list
                    for desc_amenity in desc_amenities:
                        if amenity.lower() in desc_amenity.lower():
                            amenity_found = True
                            break
                    
                    # If not found in amenities list, check in description text
                    if not amenity_found and amenity.lower() in desc_text:
                        amenity_found = True
                    
                    if amenity_found:
                        score += 1.0
            
            # Normalize score
            if fields_checked > 0:
                structured_scores[img_id] = score / fields_checked
            else:
                structured_scores[img_id] = 0.5  # Neutral score if no fields checked
        
        return structured_scores
    
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