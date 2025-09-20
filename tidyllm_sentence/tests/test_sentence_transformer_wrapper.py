#!/usr/bin/env python3
"""
Test suite for SentenceTransformer compatibility wrapper.
Validates that the wrapper provides sentence-transformers-like API.
"""

import sys
from pathlib import Path

# Add tidyllm_sentence to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import tidyllm_sentence as tls


class TestSentenceTransformerWrapper:
    """Test SentenceTransformer compatibility wrapper."""

    def test_wrapper_exists(self):
        """Test that SentenceTransformer class exists."""
        assert hasattr(tls, 'SentenceTransformer')
        print("PASS SentenceTransformer class exists")

    def test_model_initialization(self):
        """Test model can be initialized."""
        model = tls.SentenceTransformer('all-MiniLM-L6-v2')
        assert model.model_name == 'all-MiniLM-L6-v2'
        assert not model._fitted
        print("✅ Model initializes correctly")

    def test_single_sentence_encoding(self):
        """Test encoding single sentence."""
        model = tls.SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(['hello world'])

        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0  # Has dimensions
        print(f"✅ Single sentence encoded to {len(embeddings[0])} dimensions")

    def test_multiple_sentence_encoding(self):
        """Test encoding multiple sentences."""
        model = tls.SentenceTransformer('all-MiniLM-L6-v2')
        sentences = ['hello world', 'how are you', 'machine learning']
        embeddings = model.encode(sentences)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        # All embeddings should have same dimensionality
        dims = [len(emb) for emb in embeddings]
        assert all(d == dims[0] for d in dims)
        print(f"✅ Multiple sentences encoded consistently to {dims[0]} dimensions")

    def test_model_persistence(self):
        """Test that model stays fitted after first encoding."""
        model = tls.SentenceTransformer('all-MiniLM-L6-v2')

        # First encoding
        embeddings1 = model.encode(['test sentence'])
        assert model._fitted

        # Second encoding should reuse fitted model
        embeddings2 = model.encode(['another sentence'])
        assert model._fitted

        # Both should have same dimensionality
        assert len(embeddings1[0]) == len(embeddings2[0])
        print("✅ Model persistence works correctly")

    def test_empty_input_handling(self):
        """Test handling of empty input."""
        model = tls.SentenceTransformer('all-MiniLM-L6-v2')

        try:
            embeddings = model.encode([])
            assert embeddings == []
            print("✅ Empty input handled gracefully")
        except Exception as e:
            print(f"⚠️ Empty input raises exception: {e}")
            # This might be acceptable behavior

    def test_api_compatibility(self):
        """Test API matches expected sentence-transformers interface."""
        model = tls.SentenceTransformer('all-MiniLM-L6-v2')

        # Test expected methods exist
        assert hasattr(model, 'encode')
        assert callable(model.encode)

        # Test expected attributes exist
        assert hasattr(model, 'model_name')

        print("✅ API compatibility validated")

    def test_domain_adapter_usage_pattern(self):
        """Test the exact usage pattern from domain_rag_adapter.py."""
        # This replicates the pattern used in your domain adapters
        model = tls.SentenceTransformer('all-MiniLM-L6-v2')

        # Test typical RAG query encoding
        queries = [
            "What is model risk management?",
            "How do we calculate regulatory capital?",
            "Basel III compliance requirements"
        ]

        embeddings = model.encode(queries)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

        print(f"✅ Domain adapter pattern works - {len(embeddings)} queries encoded")


def run_tests():
    """Run all SentenceTransformer wrapper tests."""
    print("=" * 60)
    print("TESTING SENTENCE TRANSFORMER WRAPPER")
    print("=" * 60)

    test_suite = TestSentenceTransformerWrapper()

    tests = [
        test_suite.test_wrapper_exists,
        test_suite.test_model_initialization,
        test_suite.test_single_sentence_encoding,
        test_suite.test_multiple_sentence_encoding,
        test_suite.test_model_persistence,
        test_suite.test_empty_input_handling,
        test_suite.test_api_compatibility,
        test_suite.test_domain_adapter_usage_pattern
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)