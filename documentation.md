# Hotel Room Image Search System - Documentation

## System Overview

This system allows users to search for hotel rooms based on visual features and preferences. The system downloads hotel room images from URLs, analyzes them to extract detailed descriptions and structured features, and then enables searching through these features using natural language queries.

## System Architecture

The system consists of three main components:

1. **Image Analyzer**: Downloads hotel room images from URLs, processes them using OpenAI's GPT-4.1-mini to extract structured JSON data about room features
2. **Query Parser Agent**: Enhances natural language queries to optimize them for better keyword and semantic search performance
3. **Search Engine**: Performs keyword and semantic search to find relevant hotel rooms based on the enhanced queries

### Architecture Diagram

```
┌────────────┐     ┌────────────────┐     ┌─────────────┐     ┌─────────────┐
│            │     │                │     │             │     │             │
│  Image     │────▶│  Image         │────▶│  Downloaded │────▶│  JSON       │
│  URLs      │     │  Downloader    │     │  Images     │     │  Descriptions│
│            │     │                │     │             │     │             │
└────────────┘     └────────────────┘     └─────────────┘     └──────┬──────┘
                                                                     │
                                                                     ▼
┌──────────────┐   ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
│              │   │               │     │              │     │              │
│  User        │──▶│  Query        │────▶│  Search      │◀────│  Search      │
│  Query       │   │  Parser       │     │  Engine      │     │  Index       │
│              │   │               │     │              │     │              │
└──────────────┘   └───────────────┘     └──────┬───────┘     └──────────────┘
                                                 │
                                                 ▼
                                         ┌───────────────┐
                                         │               │
                                         │  Matching     │
                                         │  Image URLs   │
                                         │               │
                                         └───────────────┘
```

## Component Details

### 1. Image Analyzer

The Image Analyzer has two main functions:
1. **URL Processing**: Downloads images from URLs specified in a JSON file
2. **Image Analysis**: Uses OpenAI's GPT-4.1-mini model to analyze hotel room images and extract structured JSON data

It processes the following aspects of each image:

- Room Type (Single, Double, Triple, Suite, etc.)
- Maximum Capacity (number of people)
- View Type (Sea view, garden view, city view, etc.)
- Features (Desk, balcony, air conditioning, etc.)
- Room Size (Small, medium, large)
- Bed Configuration (Single beds, double beds, king size, etc.)
- Design Style (Modern, classic, minimalist, etc.)
- Extra Amenities

These features are saved as structured JSON data for each image, along with the original image URL for retrieval.

#### Robust Error Handling

The Image Analyzer includes comprehensive error handling:

1. **SSL Certificate Handling**:
   - Disables SSL verification to handle certificate issues with image URLs
   - Suppresses SSL warnings to maintain clean logs

2. **Multi-format URL Support**:
   - Tries multiple URL formats if the primary format fails
   - Handles domain variations and path differences
   - Implements retry logic with exponential backoff

3. **Incremental Processing**:
   - Checks for existing descriptions before downloading or analyzing
   - Skips already processed images to avoid redundant API calls
   - Saves progress after each image to enable resumable processing

4. **API Error Handling**:
   - Implements retry logic with exponential backoff for API rate limits
   - Properly handles various HTTP status codes
   - Differentiates between transient and permanent errors

5. **Resource Validation**:
   - Validates downloaded files before processing
   - Creates placeholder files for failed downloads
   - Checks file size to identify corrupted or incomplete downloads

### 2. Query Parser Agent

The Query Parser Agent enhances natural language queries for better search performance:

1. **Query Enhancement**:
   - Optimizes natural language queries for better keyword and semantic search
   - Identifies key search terms related to hotel room features
   - Removes filler words and adds relevant synonyms where helpful
   - Structures queries to emphasize important search terms

2. **Cost Optimization**:
   - Uses the `should_use_llm` method to determine if a query is complex enough to warrant LLM processing
   - Simple queries bypass the LLM to reduce API costs
   - Complex queries benefit from enhancement for better search accuracy

3. **Output Format**:
   - Returns both the original query and the enhanced query
   - Preserves all important constraints and requirements from the original query
   - Makes the query more effective for keyword and semantic search engines

### 3. Search Engine

The Search Engine uses two primary search approaches:

1. **Keyword-Based Search**:
   - Uses TF-IDF (Term Frequency-Inverse Document Frequency) to match keywords in descriptions
   - Efficient for simple queries with specific terms

2. **Semantic Search**:
   - Uses a sentence transformer model (all-mpnet-base-v2) for semantic understanding
   - Creates embeddings of descriptions and queries for meaning-based matching

3. **Combined Search**:
   - Integrates results from both keyword and semantic search
   - Deduplicates results, prioritizing higher scores
   - Returns a combined, ranked list of results

### 4. Main Application

Coordinates the overall workflow:
- Processes command-line arguments
- Downloads images from URLs when needed
- Triggers image analysis for downloaded images
- Initializes the search engine and query parser
- Handles user queries and displays results
- Reports performance metrics

## Data Flow

1. **Image Processing Flow**:
   - Hotel room image URLs are loaded from the `data/hotel_images.json` file
   - Images are downloaded to the `images` directory
   - The Image Analyzer processes each image using GPT-4.1-mini
   - Structured features are extracted in JSON format
   - Results are saved as JSON files in the `descriptions` directory, preserving the original URL

2. **Query Processing Flow**:
   - User submits a natural language query
   - The Query Parser Agent evaluates query complexity
   - Simple queries bypass the LLM parser
   - Complex queries are enhanced for better search performance
   - Enhanced query is passed to the Search Engine

3. **Search Flow**:
   - The Search Engine processes the enhanced query
   - Both keyword and semantic search methods are applied
   - Results are combined from both search methods
   - Matching image URLs are returned, ranked by relevance

## Technology Stack

- Python 3.10+
- OpenAI API (GPT-4.1-mini)
- Sentence Transformers (all-mpnet-base-v2 model)
- scikit-learn (for TF-IDF vectorization)
- NumPy (for vector operations)
- Requests (for downloading images)
- dotenv (for environment variable management)

## Implementation Details

### Image Analysis Process

1. **URL Download**:
   - URLs are read from the JSON file
   - Multiple URL formats are tried if the primary format fails
   - Images are downloaded and saved locally
   - The original URL is preserved for reference

2. **Image Analysis**:
   - Each image is encoded as base64
   - The encoded image is sent to OpenAI's GPT-4.1-mini with a carefully crafted system prompt
   - Low detail setting is used to reduce token usage and cost
   - The API returns structured JSON data about the hotel room
   - Both the structured data and the original image URL are saved to JSON

3. **Error Recovery**:
   - Failed downloads are marked with placeholders
   - Failed API calls are retried with exponential backoff
   - Progress is saved incrementally to allow resuming processing

### Query Enhancement Process

1. **Complexity Assessment**:
   - Query is analyzed to determine complexity
   - Checks for specific phrases that indicate complexity
   - Counts requirement indicators to assess query sophistication
   - Determines if LLM enhancement is necessary

2. **Query Optimization**:
   - Complex queries are sent to the OpenAI API
   - System prompt guides the model to enhance the query for better search
   - Identifies key search terms and removes filler words
   - Preserves all important constraints from the original query
   - Returns both original and enhanced queries

3. **Fallback Handling**:
   - If enhancement fails, falls back to the original query
   - Handles API errors gracefully
   - Ensures search can proceed even if enhancement fails

### Search Process

1. **Keyword Search**:
   - Enhanced query is converted to a TF-IDF vector
   - Cosine similarity is calculated between the query vector and all description vectors
   - Images with the highest similarity are returned

2. **Semantic Search**:
   - Enhanced query is encoded into an embedding vector using the transformer model
   - Cosine similarity is calculated between the query embedding and all description embeddings
   - Images with the highest semantic similarity are returned

3. **Result Combination**:
   - Results from both search methods are combined
   - Duplicate results are removed, keeping the highest score
   - Final results are sorted by score
   - Top results are returned to the user

## Cost Optimization

The system has been optimized for cost efficiency:

1. **Model Selection**: 
   - Using GPT-4.1-mini for image analysis (cost-effective)
   - Using local sentence transformer models for semantic search

2. **API Usage Reduction**:
   - Lower detail parameter for vision analysis
   - Reduced token generation limits
   - Efficient retries to minimize unnecessary API calls
   - LLM query enhancement only used for complex queries

3. **Query Complexity Analysis**:
   - Simple queries bypass the LLM parser completely
   - Complexity is determined by query length and specific phrases
   - Multiple requirements trigger LLM enhancement for better results
   - Optimizes cost without sacrificing search quality

4. **Performance Optimization**:
   - Local image caching to avoid repeat downloads
   - Caching embeddings for faster retrieval
   - Using vectorized operations for efficient similarity calculations
   - Incremental processing to avoid redundant work

## Performance Metrics

The system reports performance metrics to help you understand the efficiency:

- Time taken to download images from URLs
- Time taken to generate descriptions for all images
- Time taken to initialize the search engine
- Time taken to process each query
- Average query processing time

