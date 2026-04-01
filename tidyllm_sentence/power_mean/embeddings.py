"""
Power Mean Sentence Embeddings

Based on: "Concatenated Power Mean Word Embeddings as Universal
Cross-Lingual Sentence Representations" (Rücklé et al., 2018)

Generalized mean: M_p(x) = (1/n * Σ x_i^p)^(1/p)

Special cases:
- p → -∞: minimum
- p = -1: harmonic mean
- p = 0: geometric mean (limit)
- p = 1: arithmetic mean
- p = 2: quadratic mean (RMS)
- p → +∞: maximum
"""

import math
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field

Vector = List[float]
Matrix = List[Vector]

# Preset power configurations
POWER_PRESETS = {
    'default': [1, 3, float('inf')],  # Standard, emphasize-large, max
    'symmetric': [float('-inf'), 1, float('inf')],  # min, mean, max
    'comprehensive': [float('-inf'), -1, 1, 2, float('inf')],  # Full range
    'fast': [1, float('inf')],  # Just mean and max
}


@dataclass
class PowerMeanModel:
    """Power mean sentence embedding model."""
    word_vectors: Dict[str, Vector]
    embedding_dim: int  # Dimension per power mean
    powers: List[float] = field(default_factory=lambda: [1, 3, float('inf')])
    output_dim: int = 0  # Total output dimension (embedding_dim * len(powers))

    def __post_init__(self):
        self.output_dim = self.embedding_dim * len(self.powers)


def power_mean(values: List[float], p: float, eps: float = 1e-8) -> float:
    """
    Compute generalized power mean.

    M_p(x) = (1/n * Σ x_i^p)^(1/p)

    Args:
        values: List of values
        p: Power parameter
        eps: Small value to avoid division by zero

    Returns:
        Power mean value
    """
    if not values:
        return 0.0

    n = len(values)

    # Handle special cases
    if p == float('inf'):
        return max(values)
    elif p == float('-inf'):
        return min(values)
    elif abs(p) < eps:
        # Geometric mean (limit as p → 0)
        # exp(1/n * Σ log(x_i))
        log_sum = sum(math.log(max(abs(v), eps)) for v in values)
        return math.exp(log_sum / n)
    else:
        # General case: (1/n * Σ x^p)^(1/p)
        # Handle negative values carefully for odd powers
        powered_sum = 0.0
        for v in values:
            if v >= 0:
                powered_sum += v ** p
            elif p == int(p) and int(p) % 2 == 1:
                # Odd integer power preserves sign
                powered_sum += -((-v) ** p)
            else:
                # Use absolute value for non-integer or even powers
                powered_sum += (abs(v) + eps) ** p

        mean_powered = powered_sum / n

        if mean_powered >= 0:
            return mean_powered ** (1 / p)
        elif p == int(p) and int(p) % 2 == 1:
            return -((-mean_powered) ** (1 / p))
        else:
            return (abs(mean_powered) + eps) ** (1 / p)


def fit(
    sentences: List[str],
    word_vectors: Dict[str, Vector],
    powers: Optional[List[float]] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> PowerMeanModel:
    """
    Fit power mean model.

    Args:
        sentences: Corpus sentences (used for consistency, not actually needed)
        word_vectors: Pre-trained word vectors
        powers: List of power values (default: [1, 3, inf])
        tokenizer: Optional tokenizer function

    Returns:
        Fitted PowerMeanModel
    """
    if powers is None:
        powers = POWER_PRESETS['default']

    # Get embedding dimension from word vectors
    embedding_dim = len(next(iter(word_vectors.values()))) if word_vectors else 100

    return PowerMeanModel(
        word_vectors=word_vectors,
        embedding_dim=embedding_dim,
        powers=powers,
    )


def transform(
    sentences: List[str],
    model: PowerMeanModel,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Matrix:
    """
    Transform sentences to power mean embeddings.

    Args:
        sentences: Sentences to embed
        model: Fitted PowerMeanModel
        tokenizer: Optional tokenizer function

    Returns:
        Matrix of sentence embeddings (n_sentences, output_dim)
    """
    from ..utils.tokenize import word_tokenize

    if tokenizer is None:
        tokenizer = word_tokenize

    embeddings = []

    for sentence in sentences:
        tokens = tokenizer(sentence)

        if not tokens:
            embeddings.append([0.0] * model.output_dim)
            continue

        # Get word vectors for tokens
        token_vectors = []
        for token in tokens:
            if token in model.word_vectors:
                token_vectors.append(model.word_vectors[token])

        if not token_vectors:
            embeddings.append([0.0] * model.output_dim)
            continue

        # Compute power means for each dimension
        dim = model.embedding_dim
        power_embeddings = []

        for p in model.powers:
            power_emb = []
            for d in range(dim):
                values = [vec[d] for vec in token_vectors]
                pm = power_mean(values, p)
                power_emb.append(pm)
            power_embeddings.extend(power_emb)

        embeddings.append(power_embeddings)

    return embeddings


def fit_transform(
    sentences: List[str],
    word_vectors: Dict[str, Vector],
    powers: Optional[List[float]] = None,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[Matrix, PowerMeanModel]:
    """
    Fit power mean model and transform sentences.

    Args:
        sentences: Sentences to embed
        word_vectors: Pre-trained word vectors
        powers: List of power values
        tokenizer: Optional tokenizer function

    Returns:
        Tuple of (embeddings, model)
    """
    model = fit(sentences, word_vectors, powers, tokenizer)
    embeddings = transform(sentences, model, tokenizer)
    return embeddings, model


def concatenate_methods(
    sentences: List[str],
    word_vectors: Dict[str, Vector],
    methods: List[str] = ['sif', 'power_mean'],
    target_dim: Optional[int] = None,
) -> Matrix:
    """
    Concatenate embeddings from multiple methods.

    This can improve quality by combining complementary representations.

    Args:
        sentences: Sentences to embed
        word_vectors: Pre-trained word vectors
        methods: List of methods to combine ('sif', 'power_mean', 'average')
        target_dim: Target total dimension (will truncate/pad if needed)

    Returns:
        Concatenated embeddings
    """
    from ..sif.embeddings import fit_transform as sif_fit_transform

    all_embeddings = []
    dim = len(next(iter(word_vectors.values()))) if word_vectors else 100

    for method in methods:
        if method == 'sif':
            embs, _ = sif_fit_transform(sentences, word_vectors)
            all_embeddings.append(embs)

        elif method == 'power_mean':
            embs, _ = fit_transform(sentences, word_vectors)
            all_embeddings.append(embs)

        elif method == 'average':
            # Simple average
            from ..utils.tokenize import word_tokenize
            embs = []
            for sentence in sentences:
                tokens = word_tokenize(sentence)
                vecs = [word_vectors[t] for t in tokens if t in word_vectors]
                if vecs:
                    avg = [sum(v[d] for v in vecs) / len(vecs) for d in range(dim)]
                else:
                    avg = [0.0] * dim
                embs.append(avg)
            all_embeddings.append(embs)

    # Concatenate all embeddings
    result = []
    n_sentences = len(sentences)

    for i in range(n_sentences):
        concat = []
        for embs in all_embeddings:
            concat.extend(embs[i])
        result.append(concat)

    # Adjust dimension if target specified
    if target_dim is not None:
        for i in range(len(result)):
            if len(result[i]) >= target_dim:
                result[i] = result[i][:target_dim]
            else:
                result[i] = result[i] + [0.0] * (target_dim - len(result[i]))

    return result
