"""
SIF (Smooth Inverse Frequency) Sentence Embeddings

Based on: "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"
Arora, Liang, Ma (ICLR 2017)

Key insight: Frequent words dominate average embeddings but carry little meaning.
SIF down-weights frequent words and removes the common discourse vector.
"""

import math
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

Vector = List[float]
Matrix = List[Vector]


@dataclass
class SIFModel:
    """SIF sentence embedding model."""
    word_vectors: Dict[str, Vector]  # word -> embedding
    word_frequencies: Dict[str, float]  # word -> probability
    embedding_dim: int
    sif_a: float = 1e-3  # Smoothing parameter (typically 1e-3 to 1e-4)
    principal_component: Optional[Vector] = None  # First PC to remove


def fit(
    sentences: List[str],
    word_vectors: Dict[str, Vector],
    sif_a: float = 1e-3,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> SIFModel:
    """
    Fit SIF model by computing word frequencies.

    Args:
        sentences: Corpus sentences for frequency estimation
        word_vectors: Pre-trained word vectors (word -> embedding)
        sif_a: SIF smoothing parameter (default 1e-3)
        tokenizer: Optional tokenizer function

    Returns:
        Fitted SIFModel
    """
    from ..utils.tokenize import word_tokenize

    if tokenizer is None:
        tokenizer = word_tokenize

    # Compute word frequencies
    word_counts: Dict[str, int] = {}
    total_words = 0

    for sentence in sentences:
        tokens = tokenizer(sentence)
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
            total_words += 1

    # Convert to probabilities
    word_frequencies = {w: c / total_words for w, c in word_counts.items()}

    # Get embedding dimension from word vectors
    embedding_dim = len(next(iter(word_vectors.values()))) if word_vectors else 100

    model = SIFModel(
        word_vectors=word_vectors,
        word_frequencies=word_frequencies,
        embedding_dim=embedding_dim,
        sif_a=sif_a,
    )

    return model


def transform(
    sentences: List[str],
    model: SIFModel,
    remove_pc: bool = True,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Matrix:
    """
    Transform sentences to SIF embeddings.

    Args:
        sentences: Sentences to embed
        model: Fitted SIFModel
        remove_pc: Whether to remove first principal component
        tokenizer: Optional tokenizer function

    Returns:
        Matrix of sentence embeddings
    """
    from ..utils.tokenize import word_tokenize

    if tokenizer is None:
        tokenizer = word_tokenize

    embeddings = []

    for sentence in sentences:
        tokens = tokenizer(sentence)

        if not tokens:
            embeddings.append([0.0] * model.embedding_dim)
            continue

        # Compute SIF-weighted average
        weighted_sum = [0.0] * model.embedding_dim
        total_weight = 0.0

        for token in tokens:
            if token not in model.word_vectors:
                continue

            # SIF weight: a / (a + p(w))
            p_w = model.word_frequencies.get(token, 1e-9)
            weight = model.sif_a / (model.sif_a + p_w)

            word_vec = model.word_vectors[token]
            for i in range(model.embedding_dim):
                weighted_sum[i] += weight * word_vec[i]
            total_weight += weight

        # Normalize
        if total_weight > 0:
            embedding = [x / total_weight for x in weighted_sum]
        else:
            embedding = [0.0] * model.embedding_dim

        embeddings.append(embedding)

    # Remove first principal component if requested
    if remove_pc and len(embeddings) > 1:
        pc = compute_principal_component(embeddings)
        embeddings = remove_principal_component(embeddings, pc)
        model.principal_component = pc

    return embeddings


def fit_transform(
    sentences: List[str],
    word_vectors: Dict[str, Vector],
    sif_a: float = 1e-3,
    remove_pc: bool = True,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[Matrix, SIFModel]:
    """
    Fit SIF model and transform sentences in one step.

    Args:
        sentences: Sentences to embed
        word_vectors: Pre-trained word vectors
        sif_a: SIF smoothing parameter
        remove_pc: Whether to remove first principal component
        tokenizer: Optional tokenizer function

    Returns:
        Tuple of (embeddings, model)
    """
    model = fit(sentences, word_vectors, sif_a, tokenizer)
    embeddings = transform(sentences, model, remove_pc, tokenizer)
    return embeddings, model


def compute_principal_component(embeddings: Matrix, n_iter: int = 50) -> Vector:
    """
    Compute first principal component using power iteration.

    This is the "common discourse vector" that should be removed.

    Args:
        embeddings: Matrix of sentence embeddings
        n_iter: Number of power iterations

    Returns:
        First principal component vector
    """
    import random

    n_samples = len(embeddings)
    dim = len(embeddings[0]) if embeddings else 0

    if n_samples == 0 or dim == 0:
        return []

    # Center embeddings
    mean = [sum(emb[i] for emb in embeddings) / n_samples for i in range(dim)]
    centered = [[emb[i] - mean[i] for i in range(dim)] for emb in embeddings]

    # Power iteration to find first eigenvector
    rng = random.Random(42)
    pc = [rng.gauss(0, 1) for _ in range(dim)]

    # Normalize
    norm = math.sqrt(sum(x * x for x in pc))
    if norm > 0:
        pc = [x / norm for x in pc]

    for _ in range(n_iter):
        # X^T X v = sum over samples of (x . v) * x
        new_pc = [0.0] * dim

        for emb in centered:
            dot = sum(emb[i] * pc[i] for i in range(dim))
            for i in range(dim):
                new_pc[i] += dot * emb[i]

        # Normalize
        norm = math.sqrt(sum(x * x for x in new_pc))
        if norm > 1e-10:
            pc = [x / norm for x in new_pc]
        else:
            break

    return pc


def remove_principal_component(embeddings: Matrix, pc: Vector) -> Matrix:
    """
    Remove projection onto principal component from all embeddings.

    v_new = v - (v . u) * u

    Args:
        embeddings: Matrix of sentence embeddings
        pc: Principal component vector (unit norm)

    Returns:
        Embeddings with PC removed
    """
    if not pc or not embeddings:
        return embeddings

    dim = len(pc)
    result = []

    for emb in embeddings:
        # Compute projection
        dot = sum(emb[i] * pc[i] for i in range(dim))

        # Remove projection
        new_emb = [emb[i] - dot * pc[i] for i in range(dim)]
        result.append(new_emb)

    return result
