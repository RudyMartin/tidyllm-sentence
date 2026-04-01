"""
SIF (Smooth Inverse Frequency) Sentence Embeddings

Implementation of "A Simple but Tough-to-Beat Baseline for Sentence Embeddings"
(Arora et al., ICLR 2017)

Algorithm:
1. Compute weighted average: v_s = (1/|s|) * Σ (a / (a + p(w))) * v_w
2. Remove first principal component: v_s = v_s - u * u^T * v_s

This module provides SIF weighting that can work with any word vectors.
"""

from .embeddings import (
    SIFModel,
    fit,
    transform,
    fit_transform,
    compute_principal_component,
    remove_principal_component,
)

__all__ = [
    'SIFModel',
    'fit',
    'transform',
    'fit_transform',
    'compute_principal_component',
    'remove_principal_component',
]
