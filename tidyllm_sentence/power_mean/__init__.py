"""
Power Mean Sentence Embeddings

Implementation of "Concatenated Power Mean Word Embeddings as Universal
Cross-Lingual Sentence Representations" (Rücklé et al., 2018)

Key idea: Concatenate multiple power means to capture different aspects:
- p=1: Arithmetic mean (standard average)
- p=2: Root mean square (emphasizes larger values)
- p=∞: Max pooling (captures dominant features)
- p=-∞: Min pooling (captures baseline features)

The concatenation preserves more information than any single mean.
"""

from .embeddings import (
    PowerMeanModel,
    fit,
    transform,
    fit_transform,
    power_mean,
    POWER_PRESETS,
)

__all__ = [
    'PowerMeanModel',
    'fit',
    'transform',
    'fit_transform',
    'power_mean',
    'POWER_PRESETS',
]
