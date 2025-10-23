# tidyllm-sentence/__init__.py — Pure Python sentence embeddings

# TF-IDF based embeddings
from .tfidf.embeddings import fit as tfidf_fit, transform as tfidf_transform, fit_transform as tfidf_fit_transform

# Word averaging embeddings  
from .word_avg.embeddings import fit as word_avg_fit, transform as word_avg_transform, fit_transform as word_avg_fit_transform

# N-gram based embeddings
from .ngram.embeddings import fit as ngram_fit, transform as ngram_transform, fit_transform as ngram_fit_transform

# LSA/SVD based embeddings
from .lsa.embeddings import fit as lsa_fit, transform as lsa_transform, fit_transform as lsa_fit_transform

# Transformer-enhanced embeddings
from .transformer.embeddings import transformer_fit, transformer_transform, transformer_fit_transform

# Utilities
from .utils.tokenize import simple_tokenize, word_tokenize, char_ngrams
from .utils.similarity import cosine_similarity, semantic_search
from .utils.preprocessing import preprocess_for_embeddings, PreprocessingPipeline, STANDARD_PIPELINE, MINIMAL_PIPELINE, AGGRESSIVE_PIPELINE
from .utils.stemmer import simple_stem, porter_stem
from .utils.stopwords import ENGLISH_STOP_WORDS, CORE_STOP_WORDS, remove_stopwords

# Reasoning capabilities (TENSOR LOGIC ADDITION)
from .reasoning import (
    analogical_reasoning,
    case_retrieval,
    similarity_based_inference,
    temperature_sweep,
    multi_query_reasoning,
)

__version__ = "0.1.0"

# Academic Validation & Research Documentation:
# - EXHIBIT_1_ACADEMIC_BENCHMARK.md: Peer-review ready comparison vs FAISS
# - TLM_TEAM_STRATEGY.md: Multi-algorithm ensemble strategy  
# - FINAL_RABBIT_SUMMARY.md: Complete competitive analysis
# 
# The tidyllm-verse: Where algorithmic sovereignty meets research competitiveness

# Compatibility wrapper for sentence-transformers-like API
class SentenceTransformer:
    """Compatibility wrapper for sentence-transformers API."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._fitted = False
        self._vectorizer = None

    def encode(self, sentences, **kwargs):
        """Encode sentences to embeddings using TF-IDF."""
        # Handle single string input (convert to list)
        if isinstance(sentences, str):
            sentences = [sentences]
            single_input = True
        else:
            single_input = False

        if not self._fitted:
            # Fit on the input sentences
            self._vectorizer = tfidf_fit(sentences)
            self._fitted = True

        # Transform sentences to embeddings (correct argument order)
        embeddings_list = tfidf_transform(sentences, self._vectorizer)

        # Create numpy-like wrapper for compatibility
        class NumpyLikeArray:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, idx):
                return NumpyLikeEmbedding(self.data[idx])

            def __len__(self):
                return len(self.data)

            def tolist(self):
                return self.data

        class NumpyLikeEmbedding:
            def __init__(self, embedding):
                self.embedding = embedding

            def tolist(self):
                return self.embedding

            def __iter__(self):
                return iter(self.embedding)

            def __len__(self):
                return len(self.embedding)

        # Return numpy-like array wrapper
        if single_input:
            return NumpyLikeEmbedding(embeddings_list[0])
        else:
            return NumpyLikeArray(embeddings_list)

__all__ = [
    # TF-IDF
    'tfidf_fit', 'tfidf_transform', 'tfidf_fit_transform',
    # Word averaging
    'word_avg_fit', 'word_avg_transform', 'word_avg_fit_transform',
    # N-gram
    'ngram_fit', 'ngram_transform', 'ngram_fit_transform',
    # LSA
    'lsa_fit', 'lsa_transform', 'lsa_fit_transform',
    # Transformer
    'transformer_fit', 'transformer_transform', 'transformer_fit_transform',
    # Utils
    'simple_tokenize', 'word_tokenize', 'char_ngrams',
    'cosine_similarity', 'semantic_search',
    'preprocess_for_embeddings', 'PreprocessingPipeline',
    'STANDARD_PIPELINE', 'MINIMAL_PIPELINE', 'AGGRESSIVE_PIPELINE',
    'simple_stem', 'porter_stem', 'ENGLISH_STOP_WORDS', 'remove_stopwords',
    # Reasoning (TENSOR LOGIC ADDITION)
    'analogical_reasoning', 'case_retrieval', 'similarity_based_inference',
    'temperature_sweep', 'multi_query_reasoning',
    # Compatibility
    'SentenceTransformer',
]