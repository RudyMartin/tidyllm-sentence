"""
Tests for parameter validation and error handling.

These tests ensure that invalid inputs are caught early with clear error messages,
rather than causing confusing failures downstream.
"""

import pytest
import warnings
from pathlib import Path
import tempfile

import tidyllm_sentence as tls


class TestSentenceTransformerValidation:
    """Test parameter validation in SentenceTransformer."""

    def test_invalid_method_raises_valueerror(self):
        """Invalid method should raise ValueError with valid options listed."""
        with pytest.raises(ValueError) as exc_info:
            tls.SentenceTransformer(method='invalid_method')

        error_msg = str(exc_info.value)
        assert 'invalid_method' in error_msg
        assert 'lsa' in error_msg  # Should list valid options

    def test_valid_methods_accepted(self):
        """All valid methods should be accepted."""
        valid_methods = ['lsa', 'sif', 'power_mean', 'tfidf']
        for method in valid_methods:
            model = tls.SentenceTransformer(method=method)
            assert model.method == method

    def test_negative_embedding_dim_raises_valueerror(self):
        """Negative embedding_dim should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            tls.SentenceTransformer(embedding_dim=-1)

        assert '-1' in str(exc_info.value)
        assert 'positive' in str(exc_info.value).lower()

    def test_zero_embedding_dim_raises_valueerror(self):
        """Zero embedding_dim should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            tls.SentenceTransformer(embedding_dim=0)

        assert '0' in str(exc_info.value)

    def test_positive_embedding_dim_accepted(self):
        """Positive embedding_dim should work."""
        model = tls.SentenceTransformer(embedding_dim=64)
        assert model.get_sentence_embedding_dimension() == 64

    def test_default_embedding_dim_is_384(self):
        """Default embedding_dim should be 384 (MiniLM compatible)."""
        model = tls.SentenceTransformer()
        assert model.get_sentence_embedding_dimension() == 384


class TestVectorLoaderWarnings:
    """Test warning behavior for malformed vector files."""

    def test_malformed_lines_trigger_warning(self):
        """Malformed lines should trigger UserWarning when >1% skipped."""
        # Create a temp file with malformed lines
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write 50 good lines and 10 bad lines (20% bad)
            for i in range(50):
                f.write(f"word{i} " + " ".join(["0.1"] * 10) + "\n")
            for i in range(10):
                f.write(f"badline{i}\n")  # No vector, just word
            temp_path = f.name

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                vectors = tls.load_glove(temp_path)

                # Should have warning about skipped lines
                assert len(w) >= 1
                assert any('malformed' in str(warning.message).lower() or
                          'skipped' in str(warning.message).lower()
                          for warning in w)
        finally:
            Path(temp_path).unlink()

    def test_valid_file_no_warning(self):
        """Valid vector file should not trigger warnings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for i in range(100):
                f.write(f"word{i} " + " ".join(["0.1"] * 10) + "\n")
            temp_path = f.name

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                vectors = tls.load_glove(temp_path)

                # Should have no warnings about malformed lines
                malformed_warnings = [
                    warning for warning in w
                    if 'malformed' in str(warning.message).lower() or
                       'skipped' in str(warning.message).lower()
                ]
                assert len(malformed_warnings) == 0
                assert len(vectors) == 100
        finally:
            Path(temp_path).unlink()


class TestEmptyInputHandling:
    """Test handling of empty inputs."""

    def test_empty_sentence_list_returns_empty(self):
        """Empty sentence list should return empty embeddings."""
        model = tls.SentenceTransformer()
        result = model.encode([])
        assert len(result) == 0

    def test_single_sentence_returns_correct_shape(self):
        """Single sentence should return correct shape."""
        model = tls.SentenceTransformer(embedding_dim=384)
        result = model.encode(['Hello world'])
        assert len(result) == 1
        emb = result[0].tolist() if hasattr(result[0], 'tolist') else result[0]
        assert len(emb) == 384


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
