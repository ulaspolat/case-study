"""
Query Parser Agent for Hotel Room Search System

This module provides an LLM-based agent that can parse natural language queries
into structured search parameters for more accurate hotel room searches.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QueryParserAgent:
    """
    An LLM-based agent that parses natural language queries into structured search parameters.
    This agent can handle complex, varied, and natural language queries and extract structured
    information even from ambiguous or unusual phrasing.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the query parser agent with an OpenAI API key.
        
        Args:
            api_key: OpenAI API key (optional, will use environment variable if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it directly or via .env file.")
    
    def parse_query(self, query_text):
        """
        Parse a natural language query into structured search parameters.
        
        Args:
            query_text: Natural language query from the user
            
        Returns:
            Dictionary containing structured search parameters
        """
        try:
            # Prepare the system prompt for query parsing
            system_prompt = """
            You are a hotel room search query parser. Your task is to extract structured information from natural language queries
            about hotel rooms. Parse the user's query and extract the following information in JSON format:
            
            {
                "room_type": null or string (e.g., "single", "double", "triple", "suite"),
                "min_capacity": null or integer (minimum number of people the room must accommodate),
                "max_capacity": null or integer (maximum number of people the room should accommodate),
                "view_type": null or string (e.g., "sea", "garden", "city", "mountain"),
                "features": [] or list of strings (e.g., ["desk", "balcony", "air conditioning"]),
                "room_size": null or string (e.g., "small", "medium", "large"),
                "bed_configuration": null or string (e.g., "single beds", "double bed", "king size"),
                "design_style": null or string (e.g., "modern", "classic", "minimalist"),
                "extra_amenities": [] or list of strings (e.g., ["TV", "minibar", "coffee machine"]),
                "original_query": the original query text
            }
            
            Important guidelines:
            1. Capacity interpretation:
               - When a query mentions "suitable for X people" or "family of X", set min_capacity to X
               - When a query mentions "maximum of X people" or "up to X people", set max_capacity to X
               - For room types, infer capacity: single=1, double=2, triple=3, family=3+, suite=2+
            
            2. Logical operators:
               - Pay careful attention to logical operators like "and" and "or" in the query
               - When features are connected with "and" (e.g., "balcony and air conditioning"), include both in the features list
               - Make sure to include ALL required features mentioned in the query
               - The system will enforce strict "and" logic, requiring ALL features to be present
            
            3. Be precise and thorough:
               - Extract all relevant information from the query
               - Don't omit any features or requirements mentioned in the query
               - Be specific with feature names (e.g., "air conditioning" not just "air")
            
            Only include fields that are explicitly mentioned or strongly implied in the query. Use null for fields that are not mentioned.
            Your response must be valid JSON that can be parsed directly.
            """
            
            # Make API call to OpenAI
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": query_text
                    }
                ],
                "max_tokens": 500
            }
            
            # Try up to 3 times with exponential backoff
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    
                    # Check if request was successful
                    if response.status_code == 200:
                        response_data = response.json()
                        json_content = response_data['choices'][0]['message']['content']
                        
                        # Try to parse the JSON response
                        try:
                            # Clean up the response if it contains markdown code blocks
                            if json_content.startswith('```json'):
                                json_content = json_content.split('```json')[1]
                            if '```' in json_content:
                                json_content = json_content.split('```')[0]
                                
                            # Parse the JSON content
                            structured_query = json.loads(json_content.strip())
                            
                            # Add the original query
                            structured_query['original_query'] = query_text
                            
                            return structured_query
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON response: {e}")
                            # Fallback to returning a basic structure with the original query
                            return {"original_query": query_text}
                    elif response.status_code == 429 or response.status_code >= 500:
                        # Rate limit or server error, retry after delay
                        import time
                        wait_time = (2 ** retry) * 3  # Exponential backoff: 3, 6, 12 seconds
                        print(f"API rate limit or server error. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        if retry == max_retries - 1:
                            raise Exception(f"API request failed after {max_retries} retries: {response.text}")
                    else:
                        # Other error, don't retry
                        raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
                except Exception as e:
                    if retry < max_retries - 1:
                        import time
                        wait_time = (2 ** retry) * 3
                        print(f"Error during API call: {str(e)}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        raise
            
            # If we get here, all retries failed
            return {"original_query": query_text, "error": "Failed to parse query after multiple attempts"}
            
        except Exception as e:
            print(f"Error parsing query: {str(e)}")
            return {"original_query": query_text, "error": str(e)}
    
    def parse_structured_query(self, query_text):
        """
        Parse a natural language query and return structured data that can be used by the search engine.
        This is the main interface method that should be called by other components.
        
        Args:
            query_text: Natural language query from the user
            
        Returns:
            Dictionary containing the original query and structured data if parsing was successful
        """
        # Only parse complex queries
        if not self.should_use_llm(query_text):
            return {"original_query": query_text}
            
        # Parse the query to get structured information
        try:
            structured_query = self.parse_query(query_text)
            print(f"Structured query: {json.dumps(structured_query, indent=2)}")
            return structured_query
        except Exception as e:
            print(f"Error parsing query: {str(e)}")
            return {"original_query": query_text}
    
    @staticmethod
    def should_use_llm(query_text):
        """
        Determine if the query is complex enough to warrant using the LLM parser.
        This helps optimize costs by only using the LLM for complex queries.
        
        Args:
            query_text: The query text to analyze
            
        Returns:
            Boolean indicating whether to use the LLM parser
        """
        # Simple length-based heuristic
        if len(query_text.split()) > 6:  # More than 6 words suggests complexity
            return True
            
        # Check for specific phrases that indicate complexity
        complex_phrases = [
            "suitable for", "family of", "accommodate", "minimum", "maximum", "at least",
            "up to", "no more than", "with a", "that has", "featuring", "including",
            "prefer", "would like", "looking for", "need a", "want a"
        ]
        
        for phrase in complex_phrases:
            if phrase in query_text.lower():
                return True
                
        # Check for multiple requirements
        requirement_count = 0
        requirement_indicators = ["view", "balcony", "desk", "air conditioning", "bed", "room", "people", "capacity"]
        
        for indicator in requirement_indicators:
            if indicator in query_text.lower():
                requirement_count += 1
                
        return requirement_count >= 2  # Multiple requirements suggest complexity
    
    def _calculate_match_score(self, structured_query, description_data):
        """
        Calculate a match score based on how well the description matches the structured query.
        
        Args:
            structured_query: Structured query parameters
            description_data: Description data for an image
            
        Returns:
            Match score between 0 and 1
        """
        # Initialize score and count of matching fields
        score = 0
        fields_checked = 0
        
        # Check room type
        if structured_query.get('room_type') and description_data.get('room_type'):
            fields_checked += 1
            if structured_query['room_type'].lower() in description_data['room_type'].lower():
                score += 1
        
        # Check min capacity
        if structured_query.get('min_capacity') and description_data.get('max_capacity'):
            fields_checked += 1
            query_min_capacity = structured_query['min_capacity']
            desc_capacity = description_data['max_capacity']
            
            # Try to convert to integers for comparison
            try:
                if isinstance(query_min_capacity, str):
                    query_min_capacity = int(''.join(filter(str.isdigit, query_min_capacity)))
                if isinstance(desc_capacity, str):
                    desc_capacity = int(''.join(filter(str.isdigit, desc_capacity)))
                
                # Room must accommodate at least the minimum number of people
                if query_min_capacity <= desc_capacity:
                    score += 1
            except (ValueError, TypeError):
                # If conversion fails, do a string comparison
                if str(query_min_capacity) == str(desc_capacity):
                    score += 1
        
        # Check max capacity
        if structured_query.get('max_capacity') and description_data.get('max_capacity'):
            fields_checked += 1
            query_max_capacity = structured_query['max_capacity']
            desc_capacity = description_data['max_capacity']
            
            # Try to convert to integers for comparison
            try:
                if isinstance(query_max_capacity, str):
                    query_max_capacity = int(''.join(filter(str.isdigit, query_max_capacity)))
                if isinstance(desc_capacity, str):
                    desc_capacity = int(''.join(filter(str.isdigit, desc_capacity)))
                
                # Room should not exceed the maximum capacity
                if desc_capacity <= query_max_capacity:
                    score += 1
            except (ValueError, TypeError):
                # If conversion fails, do a string comparison
                if str(query_max_capacity) == str(desc_capacity):
                    score += 1
        
        # Check view type
        if structured_query.get('view_type') and description_data.get('view_type'):
            fields_checked += 1
            if structured_query['view_type'].lower() in description_data['view_type'].lower():
                score += 1
        
        # Check features
        if structured_query.get('features') and description_data.get('features'):
            query_features = structured_query['features']
            desc_features = description_data['features']
            
            # Convert to list if needed
            if isinstance(query_features, str):
                query_features = [query_features]
            if isinstance(desc_features, str):
                desc_features = [desc_features]
            
            # For 'and' logic, all features must be present
            # We'll count this as a single field check with a binary score (all or nothing)
            fields_checked += 1
            
            # Track if all features are found
            all_features_found = True
            
            # Check each feature
            for feature in query_features:
                feature_found = False
                for desc_feature in desc_features:
                    if feature.lower() in desc_feature.lower():
                        feature_found = True
                        break
                
                # If any feature is missing, the 'and' condition fails
                if not feature_found:
                    all_features_found = False
                    break
            
            # Only give a score if ALL features are found (strict 'and' logic)
            if all_features_found:
                score += 1
        
        # Check room size
        if structured_query.get('room_size') and description_data.get('room_size'):
            fields_checked += 1
            if structured_query['room_size'].lower() in description_data['room_size'].lower():
                score += 1
        
        # Check bed configuration
        if structured_query.get('bed_configuration') and description_data.get('bed_configuration'):
            fields_checked += 1
            if structured_query['bed_configuration'].lower() in description_data['bed_configuration'].lower():
                score += 1
        
        # Check design style
        if structured_query.get('design_style') and description_data.get('design_style'):
            fields_checked += 1
            if structured_query['design_style'].lower() in description_data['design_style'].lower():
                score += 1
        
        # Check extra amenities
        if structured_query.get('extra_amenities') and description_data.get('extra_amenities'):
            query_amenities = structured_query['extra_amenities']
            desc_amenities = description_data['extra_amenities']
            
            # Convert to list if needed
            if isinstance(query_amenities, str):
                query_amenities = [query_amenities]
            if isinstance(desc_amenities, str):
                desc_amenities = [desc_amenities]
            
            # Check each amenity
            for amenity in query_amenities:
                fields_checked += 1
                amenity_found = False
                for desc_amenity in desc_amenities:
                    if amenity.lower() in desc_amenity.lower():
                        amenity_found = True
                        break
                
                if amenity_found:
                    score += 1
        
        # Calculate final score
        if fields_checked > 0:
            return score / fields_checked
        else:
            return 0.5  # Neutral score if no fields were checked


# Example usage if this file is run directly
if __name__ == "__main__":
    try:
        parser = QueryParserAgent()
        
        # Test with a few example queries
        test_queries = [
            "Double rooms with a sea view",
            "Rooms with a balcony and air conditioning, with a city view",
            "Triple rooms with a desk",
            "Rooms with a maximum capacity of 4 people",
            "I need a modern suite with a king size bed and mountain view",
            "Show me rooms that have a minibar and are suitable for a family of 3"
        ]
        
        for query in test_queries:
            print(f"\nParsing query: {query}")
            result = parser.parse_query(query)
            print(json.dumps(result, indent=2))
    
    except Exception as e:
        print(f"Error: {str(e)}")
