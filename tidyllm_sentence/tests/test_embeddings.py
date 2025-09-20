import math
import tidyllm_sentence as tls

def test_tfidf_basic():
    """Test basic TF-IDF functionality."""
    sentences = [
        "the cat sat on the mat",
        "the dog ran in the park", 
        "cats and dogs are pets"
    ]
    
    embeddings, model = tls.tfidf_fit_transform(sentences)
    
    # Check shapes
    assert len(embeddings) == 3
    assert len(embeddings[0]) == model.vocab_size
    assert model.vocab_size > 0
    
    # Check that embeddings are different
    assert embeddings[0] != embeddings[1]

def test_word_avg_basic():
    """Test word averaging embeddings."""
    sentences = [
        "hello world",
        "goodbye world",
        "hello goodbye"
    ]
    
    embeddings, model = tls.word_avg_fit_transform(sentences, embedding_dim=50, seed=42)
    
    # Check shapes
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 50
    assert model.vocab_size > 0

def test_ngram_char():
    """Test character n-gram embeddings."""
    sentences = [
        "hello",
        "world", 
        "testing"
    ]
    
    embeddings, model = tls.ngram_fit_transform(sentences, n=3, ngram_type='char')
    
    # Check shapes
    assert len(embeddings) == 3
    assert len(embeddings[0]) == model.vocab_size
    
    # Should have character trigrams
    assert 'hel' in model.vocabulary or 'ell' in model.vocabulary

def test_ngram_word():
    """Test word n-gram embeddings."""
    sentences = [
        "the quick brown fox",
        "the lazy brown dog",
        "quick brown animals"
    ]
    
    embeddings, model = tls.ngram_fit_transform(sentences, n=2, ngram_type='word')
    
    # Check shapes
    assert len(embeddings) == 3
    assert len(embeddings[0]) == model.vocab_size
    
    # Should have word bigrams
    vocab_keys = list(model.vocabulary.keys())
    assert any(' ' in key for key in vocab_keys)  # Bigrams have spaces

def test_lsa_basic():
    """Test LSA embeddings."""
    sentences = [
        "computer science is great",
        "machine learning is fun",
        "data science uses computers",
        "algorithms are important"
    ]
    
    embeddings, model = tls.lsa_fit_transform(sentences, n_components=3)
    
    # Check shapes
    assert len(embeddings) == 4
    assert len(embeddings[0]) <= 3  # May be fewer if vocab is small
    assert len(model.components) > 0

def test_cosine_similarity():
    """Test cosine similarity utility."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    vec3 = [1.0, 0.0, 0.0]
    
    # Orthogonal vectors
    sim1 = tls.cosine_similarity(vec1, vec2)
    assert abs(sim1) < 1e-6
    
    # Identical vectors
    sim2 = tls.cosine_similarity(vec1, vec3)
    assert abs(sim2 - 1.0) < 1e-6

def test_semantic_search():
    """Test semantic search functionality."""
    query_emb = [1.0, 0.0, 0.0]
    corpus_embs = [
        [1.0, 0.0, 0.0],  # Identical
        [0.5, 0.5, 0.0],  # Somewhat similar
        [0.0, 1.0, 0.0],  # Orthogonal
        [0.0, 0.0, 1.0],  # Orthogonal
    ]
    
    results = tls.semantic_search(query_emb, corpus_embs, top_k=2)
    
    # Should return indices and scores
    assert len(results) == 2
    assert results[0][0] == 0  # Most similar should be index 0 (identical)
    assert results[0][1] > results[1][1]  # Scores should be descending