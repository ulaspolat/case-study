# Performance Test for Search Methods
import argparse
import time
import numpy as np
from pathlib import Path

from search_engine import SearchEngine
from vector_db import VectorDB

def test_semantic_search_performance(descriptions_dir="descriptions", vector_db_dir="vector_db", query_count=10):
    """
    Test and compare the performance of traditional vs. vector database semantic search.
    
    Args:
        descriptions_dir: Directory containing description files
        vector_db_dir: Directory for the vector database
        query_count: Number of queries to run for each test
    """
    print("\n" + "=" * 50)
    print(" Performance Test: Traditional vs. Vector DB Search ".center(50, "="))
    print("=" * 50)
    
    # Define test queries
    test_queries = [
        "Double rooms with a sea view",
        "Rooms with a balcony and air conditioning, with a city view",
        "Triple rooms with a desk",
        "Rooms with a maximum capacity of 4 people",
        "Suite with jacuzzi",
        "Modern design room with balcony",
        "Large room with mountain view",
        "Room with desk and TV",
        "Classic style room with double bed",
        "Small room with city view"
    ]
    
    # Ensure we have enough test queries
    if len(test_queries) < query_count:
        # Duplicate queries if we don't have enough
        test_queries = test_queries * (query_count // len(test_queries) + 1)
    
    # Get a subset of test queries
    selected_queries = test_queries[:query_count]
    
    # ===== Test with Traditional Search =====
    traditional_times = []
    
    # First, backup the vector database directory if it exists
    vector_db_path = Path(vector_db_dir)
    temp_dir = Path(f"{vector_db_dir}_temp")
    
    if vector_db_path.exists():
        print(f"\nTemporarily renaming vector database directory for fair comparison...")
        if temp_dir.exists():
            # If temp dir exists from a previous run, remove the current vector_db
            import shutil
            shutil.rmtree(vector_db_path)
        else:
            # Rename the directory to temporarily disable vector DB
            vector_db_path.rename(temp_dir)
    
    print("\n1. Testing traditional search performance:")
    
    # Initialize search engine without vector DB
    print("Initializing search engine with traditional search...")
    start_init = time.time()
    search_engine_traditional = SearchEngine(descriptions_dir=descriptions_dir, vector_db_dir="nonexistent_dir")
    end_init = time.time()
    print(f"Traditional search engine initialized in {end_init - start_init:.2f} seconds")
    
    # Run test queries
    for i, query in enumerate(selected_queries, 1):
        print(f"Running query {i}/{query_count}: {query}")
        start_time = time.time()
        results = search_engine_traditional.semantic_search(query, top_k=5)
        end_time = time.time()
        query_time = end_time - start_time
        traditional_times.append(query_time)
        print(f"Query processed in {query_time:.4f} seconds")
    
    # ===== Test with Vector DB Search =====
    
    # Restore the vector database directory if it was backed up
    if temp_dir.exists():
        print("\nRestoring vector database directory...")
        if vector_db_path.exists():
            import shutil
            shutil.rmtree(vector_db_path)
        temp_dir.rename(vector_db_path)
    
    print("\n2. Testing vector database search performance:")
    
    # Initialize search engine with vector DB
    print("Initializing search engine with vector database...")
    start_init = time.time()
    search_engine_vector = SearchEngine(descriptions_dir=descriptions_dir, vector_db_dir=vector_db_dir)
    end_init = time.time()
    print(f"Vector database search engine initialized in {end_init - start_init:.2f} seconds")
    
    # Run same test queries
    vector_db_times = []
    for i, query in enumerate(selected_queries, 1):
        print(f"Running query {i}/{query_count}: {query}")
        start_time = time.time()
        results = search_engine_vector.semantic_search(query, top_k=5)
        end_time = time.time()
        query_time = end_time - start_time
        vector_db_times.append(query_time)
        print(f"Query processed in {query_time:.4f} seconds")
    
    # ===== Compare Results =====
    avg_traditional = np.mean(traditional_times)
    avg_vector_db = np.mean(vector_db_times)
    speedup = avg_traditional / avg_vector_db if avg_vector_db > 0 else float('inf')
    
    print("\n" + "=" * 50)
    print(" Performance Comparison ".center(50, "="))
    print("=" * 50)
    print(f"Average traditional search time: {avg_traditional:.4f} seconds")
    print(f"Average vector database search time: {avg_vector_db:.4f} seconds")
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Compare individual query times
    print("\nQuery-by-query comparison:")
    print("-" * 50)
    print(f"{'Query':30} | {'Traditional':10} | {'Vector DB':10} | {'Speedup':10}")
    print("-" * 50)
    
    for i, (query, trad_time, vec_time) in enumerate(zip(selected_queries, traditional_times, vector_db_times), 1):
        query_short = query[:27] + "..." if len(query) > 30 else query
        individual_speedup = trad_time / vec_time if vec_time > 0 else float('inf')
        print(f"{query_short:30} | {trad_time:.4f}s | {vec_time:.4f}s | {individual_speedup:.2f}x")
    
    print("-" * 50)
    
    return {
        "traditional_times": traditional_times,
        "vector_db_times": vector_db_times,
        "avg_traditional": avg_traditional,
        "avg_vector_db": avg_vector_db,
        "speedup": speedup
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance test for semantic search methods")
    parser.add_argument("--desc_dir", type=str, default="descriptions", help="Directory with descriptions")
    parser.add_argument("--vector_db_dir", type=str, default="vector_db", help="Directory for vector database")
    parser.add_argument("--query_count", type=int, default=5, help="Number of queries to test")
    
    args = parser.parse_args()
    
    test_semantic_search_performance(
        descriptions_dir=args.desc_dir,
        vector_db_dir=args.vector_db_dir,
        query_count=args.query_count
    ) 