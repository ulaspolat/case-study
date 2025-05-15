"""
Query Parser Agent for Hotel Room Search System

This module provides an LLM-based agent that can enhance natural language queries
for better performance in keyword and semantic search.
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QueryParserAgent:
    """
    An LLM-based agent that enhances natural language queries for better search performance.
    This agent can improve ambiguous or complex queries to make them more effective for 
    keyword and semantic search engines.
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
        Enhance a natural language query for better search performance.
        
        Args:
            query_text: Natural language query from the user
            
        Returns:
            Dictionary containing original and enhanced queries
        """
        try:
            # Prepare the system prompt for query enhancement
            system_prompt = """
            You are a hotel room search query optimizer. Your task is to enhance and optimize natural language queries 
            about hotel rooms to improve keyword and semantic search performance.
            
            For the given user query:
            1. Identify key search terms related to hotel rooms (room types, views, amenities, etc.)
            2. Remove filler words and unnecessary language
            3. Add relevant synonyms for key terms where helpful
            4. Structure the query in a way that emphasizes important search terms
            5. Preserve ALL important constraints and requirements from the original query
            
            Return your response in JSON format with these fields:
            {
                "original_query": "the exact original query",
                "enhanced_query": "the optimized query for better search performance"
            }
            
            Important guidelines:
            - Ensure the enhanced query maintains ALL the requirements from the original query
            - Focus on making the query more effective for keyword and semantic search
            - Do not add requirements that weren't in the original query
            - Do not omit any important search criteria from the original query
            - Use clear, concise language that captures the semantic meaning
            
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
                            enhanced_query = json.loads(json_content.strip())
                            
                            # Ensure original_query is present
                            if "original_query" not in enhanced_query:
                                enhanced_query["original_query"] = query_text
                                
                            # Make sure enhanced_query field exists
                            if "enhanced_query" not in enhanced_query:
                                enhanced_query["enhanced_query"] = query_text
                                
                            return enhanced_query
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON response: {e}")
                            # Fallback to returning a basic structure with the original query
                            return {"original_query": query_text, "enhanced_query": query_text}
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
            return {"original_query": query_text, "enhanced_query": query_text, "error": "Failed to enhance query after multiple attempts"}
            
        except Exception as e:
            print(f"Error enhancing query: {str(e)}")
            return {"original_query": query_text, "enhanced_query": query_text, "error": str(e)}
    
    def parse_structured_query(self, query_text):
        """
        Enhance a natural language query for better search performance.
        This is the main interface method that should be called by other components.
        
        Args:
            query_text: Natural language query from the user
            
        Returns:
            Dictionary containing the original query and enhanced query if enhancement was successful
        """
        # Only enhance complex queries
        if not self.should_use_llm(query_text):
            return {"original_query": query_text, "enhanced_query": query_text}
            
        # Enhance the query
        try:
            enhanced_query = self.parse_query(query_text)
            print(f"Enhanced query: {enhanced_query['enhanced_query']}")
            return enhanced_query
        except Exception as e:
            print(f"Error enhancing query: {str(e)}")
            return {"original_query": query_text, "enhanced_query": query_text}
    
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
            print(f"\nOriginal query: {query}")
            result = parser.parse_query(query)
            print(f"Enhanced query: {result['enhanced_query']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
