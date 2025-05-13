# Hotel Room Image Search System

This system allows users to filter hotel room images based on visual preferences by performing advanced searches through image descriptions. It downloads hotel room images from URLs, analyzes them using AI, and allows searching based on room features with optimized query parsing and feature matching.

## Requirements

- Python 3.10+
- OpenAI API Key

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
- `--query`: Search query to run
- `--run_case_study`: Run all case study queries
- `--interactive`: Run in interactive mode
- `--top_k`: Number of top results to return (default: 5)
- `--method`: Search method to use (choices: `keyword`, `semantic`, `combined`; default: `combined`)

## System Architecture

The system consists of three main components:

1. **Image Analyzer**: Downloads hotel room images from URLs, processes them using OpenAI's GPT-4.1-mini to extract structured JSON data about room features
2. **Query Parser Agent**: Parses complex natural language queries into structured data for better search results
3. **Search Engine**: Performs keyword and semantic search with feature-based scoring to find relevant hotel rooms
4. **Main Application**: Coordinates the image download, analysis and search processes

## License

[MIT License](LICENSE) 
