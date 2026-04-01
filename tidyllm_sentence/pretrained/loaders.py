"""
Pre-trained Word Vector Loaders

Pure Python implementation - no external ML dependencies.
Supports GloVe, FastText, and Word2Vec text formats.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile

Vector = List[float]

# GloVe download URLs
GLOVE_URLS = {
    'glove.6B.50d': 'https://nlp.stanford.edu/data/glove.6B.zip',
    'glove.6B.100d': 'https://nlp.stanford.edu/data/glove.6B.zip',
    'glove.6B.200d': 'https://nlp.stanford.edu/data/glove.6B.zip',
    'glove.6B.300d': 'https://nlp.stanford.edu/data/glove.6B.zip',
    'glove.twitter.27B.25d': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.twitter.27B.50d': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.twitter.27B.100d': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.twitter.27B.200d': 'https://nlp.stanford.edu/data/glove.twitter.27B.zip',
}

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / '.tidyllm_sentence' / 'vectors'


def get_glove_path(name: str = 'glove.6B.100d', cache_dir: Optional[Path] = None) -> Path:
    """
    Get path to GloVe vectors file.

    Args:
        name: GloVe variant (e.g., 'glove.6B.100d')
        cache_dir: Optional cache directory

    Returns:
        Path to vectors file
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    return cache_dir / f"{name}.txt"


def download_glove(
    name: str = 'glove.6B.100d',
    cache_dir: Optional[Path] = None,
    verbose: bool = True
) -> Path:
    """
    Download GloVe vectors if not already cached.

    Args:
        name: GloVe variant (e.g., 'glove.6B.100d')
        cache_dir: Optional cache directory
        verbose: Print progress messages

    Returns:
        Path to downloaded vectors file
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = cache_dir / f"{name}.txt"

    if vectors_path.exists():
        if verbose:
            print(f"Using cached vectors: {vectors_path}")
        return vectors_path

    # Get URL
    base_name = '.'.join(name.split('.')[:2])  # e.g., 'glove.6B'
    url = GLOVE_URLS.get(name)

    if url is None:
        raise ValueError(f"Unknown GloVe variant: {name}. Available: {list(GLOVE_URLS.keys())}")

    # Download zip
    zip_path = cache_dir / f"{base_name}.zip"

    if not zip_path.exists():
        if verbose:
            print(f"Downloading {url}...")
            print("This may take a while (GloVe 6B is ~822MB)")

        urllib.request.urlretrieve(url, zip_path)

        if verbose:
            print(f"Downloaded to {zip_path}")

    # Extract specific file
    dim = name.split('.')[-1]  # e.g., '100d'
    txt_name = f"{base_name}.{dim}.txt"

    if verbose:
        print(f"Extracting {txt_name}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the file in the archive
        for member in zf.namelist():
            if member.endswith(f".{dim}.txt") or member == txt_name:
                zf.extract(member, cache_dir)
                # Rename if needed
                extracted_path = cache_dir / member
                if extracted_path != vectors_path:
                    extracted_path.rename(vectors_path)
                break
        else:
            raise ValueError(f"Could not find {txt_name} in archive")

    if verbose:
        print(f"Vectors ready: {vectors_path}")

    return vectors_path


def load_glove(
    path_or_name: str,
    max_vocab: Optional[int] = None,
    cache_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Vector]:
    """
    Load GloVe vectors from file or download by name.

    Args:
        path_or_name: Path to .txt file or GloVe name (e.g., 'glove.6B.100d')
        max_vocab: Maximum vocabulary size (None for all)
        cache_dir: Cache directory for downloads
        verbose: Print progress

    Returns:
        Dictionary mapping words to vectors
    """
    # Check if it's a name or path
    path = Path(path_or_name)

    if not path.exists():
        # Try as a GloVe name
        if path_or_name in GLOVE_URLS or path_or_name.startswith('glove.'):
            path = download_glove(path_or_name, cache_dir, verbose)
        else:
            raise FileNotFoundError(f"File not found: {path_or_name}")

    return _load_text_vectors(path, max_vocab, verbose, has_header=False)


def load_fasttext(
    path: str,
    max_vocab: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Vector]:
    """
    Load FastText vectors from .vec file.

    FastText format has a header line: num_words dim

    Args:
        path: Path to .vec file
        max_vocab: Maximum vocabulary size
        verbose: Print progress

    Returns:
        Dictionary mapping words to vectors
    """
    return _load_text_vectors(Path(path), max_vocab, verbose, has_header=True)


def load_word2vec_text(
    path: str,
    max_vocab: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Vector]:
    """
    Load Word2Vec vectors in text format.

    Assumes header line with vocab_size and dim.

    Args:
        path: Path to text file
        max_vocab: Maximum vocabulary size
        verbose: Print progress

    Returns:
        Dictionary mapping words to vectors
    """
    return _load_text_vectors(Path(path), max_vocab, verbose, has_header=True)


def _load_text_vectors(
    path: Path,
    max_vocab: Optional[int],
    verbose: bool,
    has_header: bool,
) -> Dict[str, Vector]:
    """
    Load vectors from text file.

    Format: word val1 val2 val3 ...

    Args:
        path: Path to text file
        max_vocab: Maximum vocabulary size
        verbose: Print progress
        has_header: Whether first line is header (vocab_size dim)

    Returns:
        Dictionary mapping words to vectors
    """
    vectors: Dict[str, Vector] = {}

    if verbose:
        print(f"Loading vectors from {path}...")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        # Skip header if present
        if has_header:
            header = f.readline()
            if verbose:
                parts = header.strip().split()
                if len(parts) == 2:
                    print(f"Header: {parts[0]} words, {parts[1]} dimensions")

        for i, line in enumerate(f):
            if max_vocab and i >= max_vocab:
                break

            parts = line.rstrip().split(' ')
            if len(parts) < 2:
                continue

            word = parts[0]
            try:
                vector = [float(x) for x in parts[1:]]
                vectors[word] = vector
            except ValueError:
                continue

            if verbose and (i + 1) % 100000 == 0:
                print(f"Loaded {i + 1} vectors...")

    if verbose:
        dim = len(next(iter(vectors.values()))) if vectors else 0
        print(f"Loaded {len(vectors)} vectors of dimension {dim}")

    return vectors


def create_random_vectors(
    vocabulary: List[str],
    dim: int = 100,
    seed: int = 42,
) -> Dict[str, Vector]:
    """
    Create random word vectors (Xavier initialization).

    Useful as fallback when pre-trained vectors unavailable.

    Args:
        vocabulary: List of words
        dim: Embedding dimension
        seed: Random seed

    Returns:
        Dictionary mapping words to random vectors
    """
    import random
    import math

    rng = random.Random(seed)
    bound = math.sqrt(6.0 / (len(vocabulary) + dim))

    vectors = {}
    for word in vocabulary:
        vectors[word] = [rng.uniform(-bound, bound) for _ in range(dim)]

    return vectors


def vectors_info(vectors: Dict[str, Vector]) -> Dict:
    """
    Get information about loaded vectors.

    Args:
        vectors: Word vectors dictionary

    Returns:
        Dictionary with vocab_size, dim, sample_words
    """
    if not vectors:
        return {'vocab_size': 0, 'dim': 0, 'sample_words': []}

    sample = list(vectors.keys())[:10]
    dim = len(vectors[sample[0]])

    return {
        'vocab_size': len(vectors),
        'dim': dim,
        'sample_words': sample,
    }
