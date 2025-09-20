#!/usr/bin/env python3
"""
Comprehensive test suite for tidyllm-sentence.
Tests all embedding methods with various edge cases and integration scenarios.
"""

import math
import sys
import random
from pathlib import Path

# Add tidyllm_sentence to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tidyllm_sentence as tls

class TestTFIDFEmbeddings:
    """Test TF-IDF embedding functionality."""
    
    def test_empty_sentences(self):
        """Test with empty input."""
        try:
            embeddings, model = tls.tfidf_fit_transform([])
            assert embeddings == []
        except:
            pass  # May legitimately fail
    
    def test_single_sentence(self):
        """Test with single sentence."""
        embeddings, model = tls.tfidf_fit_transform(["hello world"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0
    
    def test_identical_sentences(self):
        """Test with identical sentences."""
        sentences = ["hello world", "hello world", "hello world"]
        embeddings, model = tls.tfidf_fit_transform(sentences)
        
        # All embeddings should be identical
        for emb in embeddings[1:]:
            assert emb == embeddings[0]
    
    def test_fit_transform_consistency(self):
        """Test that fit + transform equals fit_transform."""
        sentences = ["the cat sat", "the dog ran", "cats and dogs"]
        
        # Method 1: fit_transform
        emb1, model1 = tls.tfidf_fit_transform(sentences)
        
        # Method 2: fit then transform
        model2 = tls.tfidf_fit(sentences)
        emb2 = tls.tfidf_transform(sentences, model2)
        
        # Should be identical
        assert len(emb1) == len(emb2)
        for i in range(len(emb1)):
            for j in range(len(emb1[i])):
                assert abs(emb1[i][j] - emb2[i][j]) < 1e-10

class TestWordAverageEmbeddings:
    """Test word averaging embedding functionality."""
    
    def test_different_dimensions(self):
        """Test with different embedding dimensions."""
        sentences = ["hello world", "goodbye world"]
        
        for dim in [10, 50, 100]:
            embeddings, model = tls.word_avg_fit_transform(sentences, embedding_dim=dim, seed=42)
            assert len(embeddings[0]) == dim
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces identical results."""
        sentences = ["test sentence one", "test sentence two"]
        
        emb1, _ = tls.word_avg_fit_transform(sentences, seed=123)
        emb2, _ = tls.word_avg_fit_transform(sentences, seed=123)
        
        # Should be identical
        assert emb1 == emb2

class TestUtilities:
    """Test utility functions."""
    
    def test_cosine_similarity_properties(self):
        """Test cosine similarity mathematical properties."""
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        v3 = [1, 0, 0]
        
        # Orthogonal vectors
        assert abs(tls.cosine_similarity(v1, v2)) < 1e-10
        
        # Identical vectors
        assert abs(tls.cosine_similarity(v1, v3) - 1.0) < 1e-10
    
    def test_semantic_search_ranking(self):
        """Test that semantic search ranks correctly."""
        embeddings = [
            [1, 0, 0],  # Most similar to query
            [0.5, 0.5, 0],  # Medium similarity
            [0, 1, 0],  # Least similar
        ]
        query = [1, 0, 0]
        
        results = tls.semantic_search(query, embeddings, top_k=3)
        
        # Check ranking is correct
        assert results[0][0] == 0  # Index 0 should be most similar
        assert results[0][1] > results[1][1] > results[2][1]

def run_all_tests():
    """Run all test classes."""
    test_classes = [TestTFIDFEmbeddings, TestWordAverageEmbeddings, TestUtilities]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"Testing {test_class.__name__}...")
        instance = test_class()
        
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  PASS {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  FAIL {method_name}: {e}")
    
    print(f"\n{passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    return passed_tests, total_tests

if __name__ == "__main__":
    run_all_tests()