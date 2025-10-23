# TidyLLM-Sentence Examples - Analogical Reasoning

This folder contains examples demonstrating the new analogical reasoning capabilities in tidyllm-sentence.

## What's New

TidyLLM-Sentence has been enhanced with **Tensor Logic** support - temperature-controlled analogical reasoning that builds on the math primitives from `tlm`.

### New Module: `reasoning.py`

Functions for case-based, similarity-driven reasoning:
- `analogical_reasoning()` - Find similar cases with temperature control
- `case_retrieval()` - Retrieve relevant cases from knowledge base
- `similarity_based_inference()` - Infer answers based on similarity
- `temperature_sweep()` - Analyze results across multiple temperatures
- `multi_query_reasoning()` - Combine results from multiple queries

## Examples

### 01_analogical_reasoning.py

Demonstrates core analogical reasoning capabilities:
- Basic case retrieval
- Temperature-controlled similarity matching
- Similarity-based inference with thresholds
- Temperature sweep analysis
- Multi-query reasoning with aggregation

**Run:**
```bash
cd examples
python 01_analogical_reasoning.py
```

**Key Concepts:**
- **T=0**: Only exact matches (deterministic)
- **T=1**: Standard similarity ranking
- **T>1**: More exploratory, diverse results

### 02_embedding_methods.py

Compares different embedding methods:
- TF-IDF (keyword-based)
- LSA (semantic, topic-based)
- Word-Average (simple baseline)
- Side-by-side comparisons
- Method selection guide
- Temperature + method interactions

**Run:**
```bash
cd examples
python 02_embedding_methods.py
```

**Method Guide:**
- **TF-IDF**: Best for keyword search, exact matching
- **LSA**: Best for semantic similarity (recommended default)
- **Word-Avg**: Best for fast baseline, simple cases

## Key Concepts

### Temperature Control

Temperature affects diversity vs precision in case retrieval:

```python
# Deterministic: only very similar cases
results = tls.analogical_reasoning(query, cases, temperature=0.0)

# Standard: normal similarity ranking
results = tls.analogical_reasoning(query, cases, temperature=1.0)

# Exploratory: more diverse results
results = tls.analogical_reasoning(query, cases, temperature=2.0)
```

### Multi-Query Aggregation

Combine results from multiple query perspectives:

```python
# Union: all unique cases from any query
results = tls.multi_query_reasoning(queries, cases, aggregation='union')

# Intersection: only cases matching ALL queries
results = tls.multi_query_reasoning(queries, cases, aggregation='intersection')

# Voting: average scores across queries
results = tls.multi_query_reasoning(queries, cases, aggregation='voting')
```

## Integration

This package is part of the TidyLLM Tensor Logic ecosystem:

1. **tlm** - Math primitives (similarity, temperature)
2. **tidyllm-sentence** (this package) - Embeddings + analogical reasoning
3. **tidyllm** - Orchestration + full temperature-controlled inference

## Use Cases

- **Case-Based Reasoning**: Find similar examples from past cases
- **Knowledge Retrieval**: Query knowledge bases semantically
- **Question Answering**: Infer answers from similar examples
- **Compliance Checking**: Match requirements against regulations
- **Document Search**: Find relevant documents by meaning
- **Temperature Sweeps**: Analyze sensitivity to temperature parameter

## Performance Tips

1. **Pre-compute embeddings** for repeated queries:
   ```python
   embeddings, model = tls.lsa_fit_transform(cases)
   # Reuse for multiple queries
   results = tls.analogical_reasoning(query, cases, embeddings=embeddings, model=model)
   ```

2. **Choose method wisely**:
   - Small corpus (<100 docs): Any method works
   - Large corpus: TF-IDF for speed, LSA for quality
   - Semantic understanding needed: Always use LSA

3. **Adjust top_k** based on use case:
   - Single best answer: `top_k=1`
   - Multiple perspectives: `top_k=5`
   - Temperature sweep: `top_k=3`

See the main tidyllm-sentence README for full API documentation.
