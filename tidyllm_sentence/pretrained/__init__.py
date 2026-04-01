"""
Pre-trained Word Vectors Module

Pure Python loaders for GloVe, FastText, and Word2Vec vectors.
No external dependencies (no torch, tensorflow, gensim required).

Usage:
    from tidyllm_sentence.pretrained import load_glove, load_fasttext

    # Load GloVe vectors
    vectors = load_glove("glove.6B.100d.txt")

    # Load FastText vectors
    vectors = load_fasttext("wiki-news-300d-1M.vec")

    # Use with SIF
    from tidyllm_sentence.sif import fit_transform
    embeddings, model = fit_transform(sentences, vectors)
"""

from .loaders import (
    load_glove,
    load_fasttext,
    load_word2vec_text,
    download_glove,
    get_glove_path,
    GLOVE_URLS,
)

__all__ = [
    'load_glove',
    'load_fasttext',
    'load_word2vec_text',
    'download_glove',
    'get_glove_path',
    'GLOVE_URLS',
]
