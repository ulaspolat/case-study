# Hotel Room Image Search System
import os
import argparse
import json
import time
from pathlib import Path
import urllib3

from image_analyzer import ImageAnalyzer
from search_engine import SearchEngine
from query_parser import QueryParserAgent

def main():
    """Main function to run the hotel room image search system."""
    # Suppress SSL warnings
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hotel Room Image Search System")
    parser.add_argument("--desc_dir", type=str, default="descriptions", help="Directory to load descriptions from")
    parser.add_argument("--images_dir", type=str, default="images", help="Directory to store downloaded images")
    parser.add_argument("--urls_file", type=str, default="data/hotel_images.json", help="JSON file containing image URLs")
    parser.add_argument("--vector_db_dir", type=str, default="vector_db", help="Directory to store vector database files")
    parser.add_argument("--process_images", action="store_true", help="Download and process images before search")
    parser.add_argument("--query", type=str, help="Run a specific search query")
    parser.add_argument("--run_case_study", action="store_true", help="Run all case study queries")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--method", type=str, choices=["keyword", "semantic", "combined"], 
                        default="combined", help="Search method to use")
    args = parser.parse_args()
    
    # Process images if requested
    if args.process_images:
        process_images(args.urls_file, args.images_dir, args.desc_dir)
    
    # Check if descriptions directory exists and has content
    desc_dir = Path(args.desc_dir)
    if not desc_dir.exists() or not list(desc_dir.glob("*.json")):
        print(f"No descriptions found in {args.desc_dir}. You may need to run with --process_images first.")
        return 1
    
    # Initialize search engine without API key
    print("Initializing search engine...")
    start_time = time.time()
    search_engine = SearchEngine(descriptions_dir=args.desc_dir, vector_db_dir=args.vector_db_dir)
    end_time = time.time()
    print(f"Search engine initialized in {end_time - start_time:.2f} seconds.")
    
    # Initialize query parser agent
    print("Initializing query parser agent...")
    query_parser = QueryParserAgent()
    
    # Run specific query if provided
    if args.query:
        print(f"\nRunning query: {args.query}")
        start_time = time.time()
        
        # Check if the query is complex enough to warrant using the LLM parser
        if QueryParserAgent.should_use_llm(args.query):
            print("Query is complex - using LLM parser...")
            structured_data = query_parser.parse_structured_query(args.query)
            if "enhanced_query" in structured_data:
                print(f"Enhanced query: {structured_data['enhanced_query']}")
            
            # Use the search engine with structured data
            results = search_engine.search(query=args.query, structured_data=structured_data, 
                                          method=args.method, top_k=args.top_k)
        else:
            print("Query is simple - using standard search...")
            results = search_engine.search(query=args.query, method=args.method, top_k=args.top_k)
        
        end_time = time.time()
        print(f"Query processed in {end_time - start_time:.2f} seconds.")
        print_search_results(results, search_engine)
    
    # Run case study queries
    if args.run_case_study or (not args.query and not args.interactive):
        run_case_study_queries(search_engine, query_parser, args.top_k, args.method)
    
    # Run in interactive mode
    if args.interactive:
        run_interactive_mode(search_engine, query_parser, args.top_k, args.method)
    
    return 0

def process_images(urls_file, images_dir, desc_dir):
    """Download images from URLs and process them to generate descriptions."""
    print(f"Processing images from {urls_file}...")
    try:
        # Initialize the image analyzer
        analyzer = ImageAnalyzer()
        
        # Download images from URLs and process them in one step
        print("Downloading and analyzing images...")
        analyzer.process_urls_from_json(urls_file, images_dir, desc_dir)
        results = analyzer.process_image_directory(images_dir, desc_dir)
        
        print(f"Successfully processed {len(results)} images.")
        return True
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return False

def run_interactive_mode(search_engine, query_parser, top_k=5, method="combined"):
    """Run the system in interactive mode where users can enter queries via the terminal."""
    print("\n" + "=" * 50)
    print(" Interactive Hotel Room Search ".center(50, "="))
    print("=" * 50)
    print("\nEnter 'exit', 'quit', or 'q' to exit the program.")
    print("Enter 'examples' to see example queries.")
    print("Enter 'stats' to see query statistics.")
    
    case_study_queries = [
        "Double rooms with a sea view",
        "Rooms with a balcony and air conditioning, with a city view",
        "Triple rooms with a desk",
        "Rooms with a maximum capacity of 4 people"
    ]
    
    while True:
        print("\nEnter your search query:")
        query = input("> ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting search system. Goodbye!")
            break
        
        if query.lower() == "examples":
            print("\nExample queries:")
            for i, example in enumerate(case_study_queries, 1):
                print(f"{i}. {example}")
            continue
        
        if query.lower() == "stats":
            stats = search_engine.get_query_statistics()
            print("\nQuery Statistics:")
            print(f"Total queries processed: {stats['total_queries']}")
            print(f"Unique queries: {stats['unique_queries']}")
            print(f"Cached queries: {stats['cached_queries']}")
            continue
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        print(f"\nSearching for: {query}")
        
        start_time = time.time()
        
        # Check if the query is complex enough to warrant using the LLM parser
        if QueryParserAgent.should_use_llm(query):
            print("Query is complex - using LLM parser...")
            structured_data = query_parser.parse_structured_query(query)
            if "enhanced_query" in structured_data:
                print(f"Enhanced query: {structured_data['enhanced_query']}")
            
            # Use the search engine with structured data
            results = search_engine.search(query=query, structured_data=structured_data, 
                                          method=method, top_k=top_k)
        else:
            print("Query is simple - using standard search...")
            results = search_engine.search(query=query, method=method, top_k=top_k)
        
        end_time = time.time()
        
        print(f"Query processed in {end_time - start_time:.2f} seconds.")
        print_search_results(results, search_engine)
        
        # Show similar query suggestions
        similar_queries = search_engine.suggest_similar_queries(query)
        if similar_queries:
            print("\nSimilar queries you might be interested in:")
            for i, (similar_query, score) in enumerate(similar_queries, 1):
                print(f"{i}. {similar_query} (similarity: {score:.2f})")

def print_search_results(results, search_engine):
    """Print formatted search results."""
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:")
    print("-" * 50)
    
    for i, (img_id, score, method) in enumerate(results, 1):
        print(f"{i}. Image ID: {img_id} (Score: {score:.3f}, Method: {method})")
        
        # Get URL for this image
        urls = search_engine.get_image_urls([img_id])
        if urls:
            print(f"   URL: {urls[0]}")
        
        print("-" * 50)
    
    # Print result URLs in the format required by the case study
    url_list = search_engine.get_image_urls([img_id for img_id, _, _ in results])
    print("\nResult URLs:")
    print(json.dumps(url_list, indent=2))

def run_case_study_queries(search_engine, query_parser, top_k=5, method="combined"):
    """Run all queries required by the case study."""
    case_study_queries = [
        "Double rooms with a sea view",
        "Rooms with a balcony and air conditioning, with a city view",
        "Triple rooms with a desk",
        "Rooms with a maximum capacity of 4 people"
    ]
    
    print("\n=== Case Study Queries ===")
    
    results_by_query = {}
    total_time = 0
    
    for query in case_study_queries:
        print(f"\nQuery: {query}")
        start_time = time.time()
        
        # Check if the query is complex enough to warrant using the LLM parser
        if QueryParserAgent.should_use_llm(query):
            print("Query is complex - using LLM parser...")
            structured_data = query_parser.parse_structured_query(query)
            if "enhanced_query" in structured_data:
                print(f"Enhanced query: {structured_data['enhanced_query']}")
            
            # Use the search engine with structured data
            results = search_engine.search(query=query, structured_data=structured_data, 
                                          method=method, top_k=top_k)
        else:
            print("Query is simple - using standard search...")
            results = search_engine.search(query=query, method=method, top_k=top_k)
        
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        print(f"Query processed in {query_time:.2f} seconds.")
        
        print_search_results(results, search_engine)
        
        # Store results for final output
        results_by_query[query] = search_engine.get_image_urls([img_id for img_id, _, _ in results])
    
    # Print final results in case study format
    print("\n=== Final Case Study Results ===")
    print(json.dumps(results_by_query, indent=2))
    print(f"\nTotal processing time for all queries: {total_time:.2f} seconds")
    print(f"Average query processing time: {total_time / len(case_study_queries):.2f} seconds")

if __name__ == "__main__":
    exit(main())
