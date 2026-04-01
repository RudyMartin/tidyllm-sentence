# tidyllm-sentence + yrsn Integration Guide

This document describes how to use tidyllm-sentence embeddings with yrsn's R/S/N projection system.

## Architecture Overview

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  tidyllm-sentence   │     │       yrsn          │     │       rsct          │
│  (Pure Python)      │────▶│   (Projection)      │────▶│   (Control Layer)   │
│                     │     │                     │     │                     │
│  - LSA embeddings   │     │  - HybridSimplex    │     │  - Gate evaluation  │
│  - SIF weighting    │     │    Rotor            │     │  - Route decisions  │
│  - Power mean       │     │  - MLP heads        │     │  - Certificates     │
│  - GloVe loader     │     │  - TrainedRSN       │     │                     │
│                     │     │    Projection       │     │                     │
│  Output: 384-dim    │     │  Output: R, S, N    │     │  Output: EXECUTE/   │
│  embeddings         │     │  (simplex)          │     │  REJECT/FALLBACK    │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
```

## Why This Separation?

| Component | Responsibility | Dependencies |
|-----------|---------------|--------------|
| **tidyllm-sentence** | Text → Embeddings | Pure Python (stdlib only) |
| **yrsn** | Embeddings → R/S/N | torch (neural projections) |
| **rsct** | R/S/N → Decisions | Pure Python |

This separation allows:
- **Offline/embedded use**: tidyllm-sentence works without ML frameworks
- **Educational transparency**: All algorithms are readable Python
- **Production quality**: yrsn's trained projections when accuracy matters

## Installation

```bash
# Core packages
pip install tidyllm-sentence  # Pure Python embeddings
pip install yrsn              # R/S/N projection (requires torch)

# Optional: swarm-it-adk adapter
pip install swarm-it-adk
```

## Usage Patterns

### Pattern 1: Direct Integration (tidyllm-sentence + yrsn)

```python
from tidyllm_sentence import SentenceTransformer
from yrsn.core.decomposition import TrainedRSNProjection

# Step 1: Generate embeddings with tidyllm-sentence
model = SentenceTransformer(embedding_dim=384, method='lsa')
texts = ["Machine learning is transforming industries", "Deep learning requires large datasets"]
embeddings = model.encode(texts)

# Convert to numpy
import numpy as np
embeddings_np = np.array(embeddings.tolist(), dtype=np.float32)

# Step 2: Project to R/S/N with yrsn
projection = TrainedRSNProjection.from_checkpoint("path/to/checkpoint.pt")
rsn_scores = projection.compute_rsn_batch(embeddings_np)

print(f"R (Relevance): {rsn_scores['R']}")
print(f"S (Stability): {rsn_scores['S']}")
print(f"N (Noise):     {rsn_scores['N']}")
```

### Pattern 2: Using swarm-it-adk Adapter

```python
from swarm_it.providers.embedding import TidyLLMSentenceProvider, check_kappa

# Create provider
provider = TidyLLMSentenceProvider(embedding_dim=384, method='lsa')

# Generate embeddings
texts = ["Example text 1", "Example text 2", "Example text 3"]
embeddings = provider.embed(texts)

# Check kappa viability for geometric operations
result = check_kappa(embeddings)
print(f"κ = {result.kappa:.2f}, viable: {result.is_viable}")

if not result.is_viable:
    print(f"Recommend expansion factor k={result.recommended_k}")
```

### Pattern 3: With Pre-trained GloVe Vectors (Higher Quality)

```python
from tidyllm_sentence import SentenceTransformer, load_glove, sif_fit_transform

# Load GloVe vectors (downloads on first use, ~800MB for 300d)
word_vectors = load_glove('glove.6B.100d')

# Option A: SIF embeddings directly
texts = ["Hello world", "Machine learning"]
embeddings, model = sif_fit_transform(texts, word_vectors)

# Option B: Via SentenceTransformer wrapper
from swarm_it.providers.embedding import TidyLLMSentenceProvider

provider = TidyLLMSentenceProvider(
    embedding_dim=384,
    method='sif',
    glove_path='glove.6B.100d',  # Will auto-download
)
embeddings = provider.embed(texts)
```

### Pattern 4: Full Pipeline (Embeddings → R/S/N → Certificate)

```python
from tidyllm_sentence import SentenceTransformer
from yrsn.core.decomposition import TrainedRSNProjection
from rsct import RSCTCertificate, evaluate_gates, route_decision

# Step 1: Embeddings
model = SentenceTransformer(embedding_dim=384)
embedding = model.encode("Analyze this text for quality")

# Step 2: R/S/N decomposition
projection = TrainedRSNProjection.from_checkpoint("checkpoint.pt")
rsn = projection.compute_rsn(embedding.tolist())

# Step 3: Create certificate
certificate = RSCTCertificate(
    R=rsn['R'],
    S=rsn['S'],
    N=rsn['N'],
    alpha=rsn['R'] / (rsn['R'] + rsn['N']),
    kappa=0.8,
    certificate_id="tidyllm-test-001",
)

# Step 4: Evaluate gates and route
certificate = evaluate_gates(certificate)
route = route_decision(certificate)

print(f"Outcome: {certificate.final_outcome}")
print(f"Should execute: {route.should_execute}")
```

## Quality Comparison

| Method | MAP Score | Memory | Dependencies |
|--------|-----------|--------|--------------|
| sentence-transformers (all-MiniLM-L6-v2) | ~85% | 88.7 MB | torch, transformers |
| tidyllm-sentence (LSA) | ~65% | 0.5 MB | None (stdlib) |
| tidyllm-sentence (SIF + GloVe) | ~70% | ~100 MB | None + GloVe file |
| tidyllm-sentence (Power Mean + GloVe) | ~72% | ~100 MB | None + GloVe file |

## When to Use tidyllm-sentence

✅ **Good for:**
- Educational/demo environments
- Embedded systems without GPU
- Air-gapped deployments
- Cost-sensitive batch processing
- Algorithmic transparency requirements

❌ **Not recommended for:**
- Production systems requiring >80% accuracy
- Real-time inference with strict latency
- Tasks where embedding quality is critical

## Kappa Viability

For geometric operations (rotors, merging), check embedding viability:

```python
from swarm_it.providers.embedding import check_kappa

result = check_kappa(embeddings)

if result.kappa < 50:
    print(f"Warning: κ={result.kappa:.1f} < 50")
    print(f"Embeddings may not have 'room to rotate'")
    print(f"Consider expansion factor k={result.recommended_k}")
```

## yrsn Projection Methods

yrsn provides multiple projection approaches:

| Method | Class | Description |
|--------|-------|-------------|
| **HybridSimplexRotor** | Geometric | Patented (P17), guarantees R+S+N=1 |
| **LearnedYRSNProjectionHeads** | MLP | Three separate heads for R, S, N |
| **TrainedRSNProjection** | Unified | Loads from checkpoint, auto-detects mode |
| **ProjectionLayer** | Linear | Simple 1024→64 compression (P18) |

For most use cases, use `TrainedRSNProjection.from_checkpoint()`.

## Troubleshooting

### Import Error: tidyllm-sentence not found
```bash
pip install tidyllm-sentence
```

### Import Error: yrsn not found
```bash
pip install yrsn  # Requires torch
```

### Dimension mismatch
Ensure `embedding_dim` matches your yrsn checkpoint:
```python
# Check checkpoint expected dimension
import torch
ckpt = torch.load("checkpoint.pt", weights_only=True)
print(f"Expected input dim: {ckpt.get('config', {}).get('input_dim', 'unknown')}")

# Match in tidyllm-sentence
model = SentenceTransformer(embedding_dim=384)  # Match this
```

### Low kappa warning
If κ < 50, your embedding space may be over-concentrated:
1. Use more diverse training texts
2. Apply capacity expansion (k factor)
3. Consider higher-dimension embeddings

## References

- [SIF Paper (ICLR 2017)](https://openreview.net/pdf?id=SyK00v5xx)
- [Power Mean Embeddings (arXiv:1803.01400)](https://arxiv.org/abs/1803.01400)
- [YRSN Theory](https://github.com/NextShiftConsulting/yrsn)
- [RSCT Control Layer](https://github.com/NextShiftConsulting/rsct)
