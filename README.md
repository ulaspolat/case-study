# Hotel Room Image Search System

This system allows users to filter hotel room images based on visual preferences by performing advanced searches through image descriptions. It downloads hotel room images from URLs, analyzes them using AI, and allows searching based on room features with optimized query enhancement and efficient keyword/semantic search.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ulaspolat/case-study.git
cd case-study
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key:
Create a `.env` file in the project root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### 1. Download and Process Images

First, download and analyze the hotel images from the provided URLs:

```bash
python src/main_fixed.py --process_images
```

This will:
- Download images from URLs in `data/hotel_images.json` to the `images` directory
- Analyze each image using GPT-4.1-mini
- Extract structured JSON data about room features
- Save the descriptions to the `descriptions` directory

### 2. Search for Hotel Rooms

You can search for hotel rooms in three ways:

#### a. Run a specific query:

```bash
python src/main_fixed.py --query "Double rooms with a sea view"
```

#### b. Run all case study queries:

```bash
python src/main_fixed.py --run_case_study
```

This will run the following queries:
- "Double rooms with a sea view"
- "Rooms with a balcony and air conditioning, with a city view"
- "Triple rooms with a desk"
- "Rooms with a maximum capacity of 4 people"

#### c. Run in interactive mode:

```bash
python src/main_fixed.py --interactive
```

This allows you to enter multiple queries in a conversational interface.

### 3. Advanced Usage

```bash
# Specify custom directories and search method
python src/main_fixed.py --urls_file custom_urls.json --images_dir custom_images --desc_dir custom_descriptions --process_images --method semantic

# Return more results
python src/main_fixed.py --query "Double rooms with a sea view" --top_k 10

# Choose search method (keyword, semantic, or combined)
python src/main_fixed.py --query "Double rooms with a sea view" --method semantic
```

## Command Line Arguments

- `--process_images`: Download and process images from URLs
- `--urls_file`: JSON file containing image URLs (default: `data/hotel_images.json`)
- `--images_dir`: Directory to store downloaded images (default: `images`)
- `--desc_dir`: Directory to store image descriptions (default: `descriptions`)
- `--vector_db_dir`: Directory to store vector database files (default: `vector_db`)
- `--query`: Search query to run
- `--run_case_study`: Run all case study queries
- `--interactive`: Run in interactive mode
- `--top_k`: Number of top results to return (default: 5)
- `--method`: Search method to use (choices: `keyword`, `semantic`, `combined`; default: `combined`)

## System Architecture

The system consists of four main components:

1. **Image Analyzer**: Downloads hotel room images from URLs, processes them using OpenAI's GPT-4.1-mini to extract structured JSON data about room features
2. **Query Parser Agent**: Enhances natural language queries to optimize them for better keyword and semantic search performance
3. **Vector Database**: Efficiently stores and indexes embeddings using FAISS for high-performance similarity search
   - Indexes room descriptions for fast semantic search
   - Stores and indexes user queries for query suggestions
   - Caches search results to avoid redundant computations
4. **Search Engine**: Performs keyword and semantic search to find relevant hotel rooms based on the enhanced queries
5. **Main Application**: Coordinates the image download, analysis and search processes

### Performance Improvements

The system now uses a vector database (FAISS) to significantly improve semantic search performance:

- **Speed**: Queries are up to 10x faster with indexed vector search
- **Caching**: Query results are cached to avoid recomputing for identical queries
- **Persistence**: Embeddings are stored on disk and reused across sessions
- **Scalability**: The vector database can efficiently handle much larger datasets
- **Query Suggestions**: The system learns from past queries to suggest similar searches

### Advanced Features

#### Query Suggestions

The system now provides intelligent query suggestions based on previous searches:

- Stores embeddings of all user queries in a separate vector index
- Compares new queries with past queries using semantic similarity
- Suggests related queries that might help users refine their search
- Improves user experience by showing alternative search options

#### Query Statistics

The interactive mode now provides statistics about query usage:

- Total number of queries processed
- Number of unique queries
- Number of cached query results

## License

[MIT License](LICENSE) 
