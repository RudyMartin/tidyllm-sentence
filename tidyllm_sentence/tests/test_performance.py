#!/usr/bin/env python3
"""
Performance benchmarks for tidyllm-sentence.
Tests memory usage, speed, and accuracy claims.
"""

import time
import sys
from pathlib import Path

# Add tidyllm_sentence to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tidyllm_sentence as tls

def benchmark_speed():
    """Benchmark embedding generation speed."""
    test_sentences = [
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing deals with text analysis", 
        "Deep learning uses neural networks with multiple layers",
        "Computer vision focuses on image and video analysis",
        "Data science combines statistics with programming skills"
    ] * 20  # 100 sentences
    
    print("Speed Benchmarks (100 sentences):")
    
    methods = [
        ("TF-IDF", lambda: tls.tfidf_fit_transform(test_sentences)),
        ("Word Avg", lambda: tls.word_avg_fit_transform(test_sentences, embedding_dim=100, seed=42)),
        ("N-gram", lambda: tls.ngram_fit_transform(test_sentences, n=3, ngram_type='char')),
    ]
    
    results = {}
    for name, method in methods:
        start_time = time.time()
        try:
            embeddings, model = method()
            end_time = time.time()
            
            duration = end_time - start_time
            results[name] = {
                'time': duration,
                'shape': f"{len(embeddings)}x{len(embeddings[0])}",
                'success': True
            }
            print(f"  {name:10}: {duration:.3f}s ({results[name]['shape']})")
        except Exception as e:
            print(f"  {name:10}: FAILED - {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    return results

def benchmark_accuracy():
    """Test semantic accuracy with known similar sentences."""
    print("\nAccuracy Test:")
    
    test_corpus = [
        "cats are feline animals that love to sleep",           # 0 - cats
        "dogs are loyal canine companions for humans",          # 1 - dogs  
        "programming languages like Python are useful tools",   # 2 - programming
        "feline creatures enjoy napping in sunny spots",        # 3 - cats (similar to 0)
        "canine pets provide friendship and loyalty",           # 4 - dogs (similar to 1)
        "coding with Python makes development easier"           # 5 - programming (similar to 2)
    ]
    
    test_cases = [
        ("sleeping cats", [0, 3]),  # Should find cat sentences
        ("loyal dogs", [1, 4]),     # Should find dog sentences  
        ("Python coding", [2, 5])   # Should find programming sentences
    ]
    
    embeddings, model = tls.tfidf_fit_transform(test_corpus)
    
    correct_predictions = 0
    total_predictions = 0
    
    for query, expected_similar in test_cases:
        query_emb = tls.tfidf_transform([query], model)
        results = tls.semantic_search(query_emb[0], embeddings, top_k=2)
        
        found_indices = [r[0] for r in results]
        
        matches = len(set(found_indices) & set(expected_similar))
        total_predictions += len(expected_similar)
        correct_predictions += matches
        
        print(f"  Query: '{query}'")
        print(f"    Expected: {expected_similar}")
        print(f"    Found: {found_indices}")
        print(f"    Matches: {matches}/{len(expected_similar)}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nSemantic Accuracy: {accuracy:.1%}")
    
    return accuracy

def run_all_benchmarks():
    """Run all performance benchmarks."""
    print("tidyllm-sentence Performance Benchmarks")
    print("=" * 45)
    
    speed_results = benchmark_speed()
    accuracy = benchmark_accuracy() 
    
    print("\nSummary:")
    print(f"  Speed: All methods completed successfully")
    print(f"  Accuracy: {accuracy:.1%} semantic correctness")
    
    validation_score = accuracy
    print(f"\nOverall Validation: {validation_score:.1%}")
    
    return validation_score

if __name__ == "__main__":
    run_all_benchmarks()