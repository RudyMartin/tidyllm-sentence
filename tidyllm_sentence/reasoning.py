"""
Reasoning capabilities using sentence embeddings.

Provides analogical reasoning, case-based retrieval, and similarity-based inference.
Uses temperature control for exploration vs exploitation tradeoff.
"""

try:
    import tlm
    TLM_AVAILABLE = True
except ImportError:
    TLM_AVAILABLE = False

from .tfidf.embeddings import fit_transform as tfidf_fit_transform, transform as tfidf_transform
from .lsa.embeddings import fit_transform as lsa_fit_transform, transform as lsa_transform
from .word_avg.embeddings import fit_transform as word_avg_fit_transform
from .utils.similarity import semantic_search


def analogical_reasoning(query, cases, embeddings=None, model=None, top_k=5, temperature=1.0, method='lsa'):
    """Find similar cases via semantic similarity with temperature control.

    Temperature controls diversity:
    - T=0: Only exact matches (similarity ≈ 1.0)
    - T=1: Standard ranking by similarity
    - T>1: More diverse results (exploration)

    Args:
        query: Query text (string)
        cases: List of case texts
        embeddings: Pre-computed case embeddings (optional)
        model: Pre-fitted model for embedding query (required if embeddings provided)
        top_k: Number of similar cases to return
        temperature: Float >= 0 controlling diversity
        method: Embedding method ('tfidf', 'lsa', 'word_avg')

    Returns:
        List of (case_idx, similarity_score, case_text) tuples

    Examples:
        >>> cases = ["Data validation is required", "Schema checks are needed"]
        >>> query = "How to validate data?"
        >>> results = analogical_reasoning(query, cases, top_k=2, temperature=1.0)
        >>> results[0][0]  # Index of most similar case
        0
    """
    # Generate embeddings if not provided (compute on-the-fly)
    if embeddings is None:
        if method == 'tfidf':
            # TF-IDF: Term frequency-inverse document frequency
            embeddings, model = tfidf_fit_transform(cases)
            query_emb = tfidf_transform([query], model)

        elif method == 'lsa':
            # LSA: Latent Semantic Analysis (dimensionality reduction via SVD)
            # Use min(50, len(cases)) to avoid over-dimensioned spaces
            embeddings, model = lsa_fit_transform(cases, n_components=min(50, len(cases)))
            query_emb = lsa_transform([query], model)

        elif method == 'word_avg':
            # Word averaging: Simple baseline using averaged word vectors
            embeddings, model = word_avg_fit_transform(cases, embedding_dim=100)
            # For word_avg, need to include query in corpus (limitation of current API)
            all_embeddings, _ = word_avg_fit_transform(cases + [query], embedding_dim=100)
            query_emb = [all_embeddings[-1]]  # Extract query embedding
            embeddings = all_embeddings[:-1]  # Extract case embeddings

        else:
            raise ValueError(f"Unknown method: {method}")

    else:
        # Use provided pre-computed embeddings (more efficient for repeated queries)
        if model is None:
            raise ValueError("Must provide model when using pre-computed embeddings")

        # Transform query using the pre-fitted model
        if method == 'tfidf':
            query_emb = tfidf_transform([query], model)
        elif method == 'lsa':
            query_emb = lsa_transform([query], model)
        else:
            raise ValueError(f"Cannot transform query for method: {method}")

    # Find k most similar cases using cosine similarity
    # Returns list of (index, similarity_score) tuples
    results = semantic_search(query_emb[0], embeddings, top_k=top_k)

    # Apply temperature scaling to control diversity vs precision
    if temperature < 1e-8:
        # T≈0: Deterministic - only exact matches (similarity very close to 1.0)
        results = [(idx, score) for idx, score in results if score > 0.999]

    elif temperature != 1.0 and TLM_AVAILABLE:
        # Apply temperature scaling to similarity scores
        # T > 1: Reduces differences (more diverse results)
        # T < 1: Increases differences (more selective)
        scores = [score for idx, score in results]
        scaled_scores = tlm.apply_temperature(scores, temperature)

        # Reconstruct results with scaled scores
        results = [(idx, scaled_scores[i]) for i, (idx, _) in enumerate(results)]

        # Re-sort by scaled scores (ranking may change with temperature)
        results.sort(key=lambda x: x[1], reverse=True)

    # Augment results with actual case text for convenience
    # Returns (index, score, text) tuples
    return [(idx, score, cases[idx]) for idx, score in results]


def case_retrieval(query, case_base, method='lsa', n_components=50, top_k=None):
    """Retrieve relevant cases from case base.

    Args:
        query: Query text
        case_base: List of case texts
        method: Embedding method ('tfidf', 'lsa', 'word_avg')
        n_components: For LSA, number of components
        top_k: Number of cases to return (None = all)

    Returns:
        List of (case_text, similarity_score) tuples sorted by relevance

    Examples:
        >>> cases = ["Python is a language", "JavaScript is a language"]
        >>> query = "What programming languages exist?"
        >>> results = case_retrieval(query, cases, method='tfidf', top_k=2)
        >>> len(results)
        2
    """
    # Default to returning all cases if top_k not specified
    if top_k is None:
        top_k = len(case_base)

    # Generate embeddings based on selected method
    if method == 'lsa':
        # LSA: Dimensionality reduction for semantic similarity
        embeddings, model = lsa_fit_transform(case_base, n_components=min(n_components, len(case_base)))
        query_emb = lsa_transform([query], model)

    elif method == 'tfidf':
        # TF-IDF: Statistical word importance
        embeddings, model = tfidf_fit_transform(case_base)
        query_emb = tfidf_transform([query], model)

    elif method == 'word_avg':
        # Word averaging: Simple mean of word vectors
        # For word_avg, need to use combined corpus (API limitation)
        embeddings_with_query, _ = word_avg_fit_transform(case_base + [query], embedding_dim=100)
        query_emb = [embeddings_with_query[-1]]  # Last embedding is query
        embeddings = embeddings_with_query[:-1]  # Rest are cases

    else:
        raise ValueError(f"Unknown method: {method}")

    # Find k most similar cases via cosine similarity
    results = semantic_search(query_emb[0], embeddings, top_k=top_k)

    # Return (case_text, similarity_score) tuples for easy access
    return [(case_base[idx], score) for idx, score in results]


def similarity_based_inference(query, knowledge_base, threshold=0.5, method='lsa'):
    """Infer answer based on similarity to knowledge base.

    Retrieves similar knowledge items and returns them if above threshold.

    Args:
        query: Query text
        knowledge_base: List of knowledge texts
        threshold: Minimum similarity score (0-1)
        method: Embedding method

    Returns:
        Dict with:
        - 'matches': List of (text, score) tuples above threshold
        - 'best_match': Highest scoring match (or None)
        - 'confidence': Score of best match

    Examples:
        >>> kb = ["The sky is blue", "Grass is green"]
        >>> query = "What color is the sky?"
        >>> result = similarity_based_inference(query, kb, threshold=0.3)
        >>> result['best_match'][0]
        'The sky is blue'
    """
    # Retrieve all cases with similarity scores
    results = case_retrieval(query, knowledge_base, method=method, top_k=len(knowledge_base))

    # Filter to keep only matches above the similarity threshold
    # Higher threshold = more selective (stricter matching)
    matches = [(text, score) for text, score in results if score >= threshold]

    # Extract the best (highest scoring) match
    best_match = matches[0] if matches else None
    confidence = best_match[1] if best_match else 0.0

    return {
        'matches': matches,           # All matches above threshold
        'best_match': best_match,     # (text, score) of top match
        'confidence': confidence,     # Score of best match
        'num_matches': len(matches)   # Count of matches above threshold
    }


def temperature_sweep(query, cases, temperatures=None, method='lsa', top_k=3):
    """Run analogical reasoning across multiple temperatures.

    Useful for understanding how temperature affects results.

    Args:
        query: Query text
        cases: List of case texts
        temperatures: List of temperature values (default: [0.0, 0.5, 1.0, 2.0])
        method: Embedding method
        top_k: Number of results per temperature

    Returns:
        Dict mapping temperature -> results

    Examples:
        >>> cases = ["Case 1", "Case 2", "Case 3"]
        >>> query = "Query"
        >>> sweep = temperature_sweep(query, cases, temperatures=[0.0, 1.0])
        >>> len(sweep)
        2
    """
    # Use default temperature range if not specified
    # Covers: deterministic (0.0), conservative (0.5), standard (1.0), exploratory (2.0)
    if temperatures is None:
        temperatures = [0.0, 0.5, 1.0, 2.0]

    # Pre-compute embeddings once for efficiency (shared across all temperatures)
    if method == 'tfidf':
        embeddings, model = tfidf_fit_transform(cases)
    elif method == 'lsa':
        embeddings, model = lsa_fit_transform(cases, n_components=min(50, len(cases)))
    elif method == 'word_avg':
        embeddings, model = word_avg_fit_transform(cases, embedding_dim=100)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Run analogical reasoning at each temperature value
    results = {}
    for temp in temperatures:
        # Use pre-computed embeddings to avoid redundant computation
        temp_results = analogical_reasoning(
            query=query,
            cases=cases,
            embeddings=embeddings,  # Reuse embeddings
            model=model,             # Reuse model
            top_k=top_k,
            temperature=temp,
            method=method
        )
        # Store results keyed by temperature
        results[temp] = temp_results

    return results


def multi_query_reasoning(queries, cases, method='lsa', top_k=3, aggregation='union'):
    """Perform reasoning over multiple queries.

    Useful for complex questions that can be decomposed.

    Args:
        queries: List of query strings
        cases: List of case texts
        method: Embedding method
        top_k: Number of results per query
        aggregation: How to combine results ('union', 'intersection', 'voting')

    Returns:
        Dict with:
        - 'per_query': Results for each query
        - 'aggregated': Combined results based on aggregation strategy

    Examples:
        >>> queries = ["What is validation?", "How to check data?"]
        >>> cases = ["Data validation required", "Schema checks needed"]
        >>> result = multi_query_reasoning(queries, cases, top_k=2)
        >>> 'per_query' in result
        True
    """
    # Run case retrieval for each query independently
    per_query_results = {}
    for query in queries:
        results = case_retrieval(query, cases, method=method, top_k=top_k)
        per_query_results[query] = results

    # Aggregate results using specified strategy
    if aggregation == 'union':
        # Combine all unique cases from all queries (broadest results)
        seen = set()
        aggregated = []
        for query, results in per_query_results.items():
            for case, score in results:
                if case not in seen:
                    seen.add(case)
                    aggregated.append((case, score))

    elif aggregation == 'intersection':
        # Only keep cases that appear in ALL query results (strictest results)
        all_cases = [set(case for case, _ in results)
                     for results in per_query_results.values()]
        common_cases = set.intersection(*all_cases) if all_cases else set()

        # Use scores from first query for the common cases
        first_results = list(per_query_results.values())[0]
        aggregated = [(case, score) for case, score in first_results
                      if case in common_cases]

    elif aggregation == 'voting':
        # Average similarity scores across all queries where case appears
        # Cases appearing in more queries with high scores rank higher
        case_scores = {}
        for query, results in per_query_results.items():
            for case, score in results:
                if case not in case_scores:
                    case_scores[case] = []
                case_scores[case].append(score)

        # Compute average score for each case and sort by score
        aggregated = [(case, sum(scores)/len(scores))
                      for case, scores in case_scores.items()]
        aggregated.sort(key=lambda x: x[1], reverse=True)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return {
        'per_query': per_query_results,
        'aggregated': aggregated
    }
