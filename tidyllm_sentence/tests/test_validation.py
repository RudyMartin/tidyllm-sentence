#!/usr/bin/env python3
"""
Validation tests for tidyllm-sentence core functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tidyllm_sentence as tls

def test_core_functionality():
    """Test that all core methods work."""
    sentences = ["The cat sat", "Dogs run", "Machine learning"]
    
    # TF-IDF
    embeddings, model = tls.tfidf_fit_transform(sentences)
    assert len(embeddings) == 3
    assert len(embeddings[0]) > 0
    
    # Word averaging
    word_embs, _ = tls.word_avg_fit_transform(sentences, embedding_dim=50, seed=42)
    assert len(word_embs) == 3
    assert len(word_embs[0]) == 50
    
    # Semantic search
    query = "cats"
    query_emb = tls.tfidf_transform([query], model)
    results = tls.semantic_search(query_emb[0], embeddings, top_k=2)
    assert len(results) == 2
    assert results[0][0] in [0, 1, 2]  # Valid index
    
    print("✅ All core functionality tests passed")

def test_performance_claims():
    """Validate key performance claims."""
    import time
    
    # Test speed claim
    test_sentences = ["Machine learning"] * 100
    start_time = time.time()
    embeddings, model = tls.tfidf_fit_transform(test_sentences)
    duration = time.time() - start_time
    
    assert duration < 1.0, f"Speed test failed: {duration:.3f}s > 1.0s"
    print(f"✅ Speed test passed: {duration:.3f}s for 100 sentences")
    
    # Test accuracy with known similar sentences
    corpus = [
        "cats are feline animals",
        "dogs are canine pets", 
        "programming with Python"
    ]
    
    embeddings, model = tls.tfidf_fit_transform(corpus)
    query_emb = tls.tfidf_transform(["feline cats"], model)
    results = tls.semantic_search(query_emb[0], embeddings, top_k=1)
    
    # Should find the cat sentence as most similar
    assert results[0][0] == 0, "Semantic accuracy test failed"
    print("✅ Semantic accuracy test passed")

if __name__ == "__main__":
    test_core_functionality()
    test_performance_claims()
    print("\n🎉 All validation tests passed!")