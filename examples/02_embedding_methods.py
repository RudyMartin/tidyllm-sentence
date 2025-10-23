"""
Example 2: Comparing Embedding Methods

Demonstrates different embedding methods (TF-IDF, LSA, Word-Avg) for reasoning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tidyllm_sentence as tls

print("=" * 60)
print("TidyLLM-Sentence: Embedding Methods Comparison")
print("=" * 60)

# Sample case base about software development
cases = [
    "Unit tests verify individual functions work correctly",
    "Integration tests check that components work together",
    "Code reviews improve code quality and catch bugs",
    "Continuous integration automates testing and deployment",
    "Version control tracks changes to code over time",
    "Documentation helps developers understand the system",
    "Static analysis detects potential bugs without running code",
    "Performance testing ensures the system meets speed requirements"
]

query = "How to ensure code quality?"

# Example 1: TF-IDF Method
print("\n1. TF-IDF Method")
print("-" * 40)
print("(Term Frequency-Inverse Document Frequency)")
print("Best for: Keyword-based matching, document retrieval")

results_tfidf = tls.case_retrieval(query, cases, method='tfidf', top_k=3)

print(f"\nQuery: {query}")
print("\nTop 3 results:")
for i, (case_text, score) in enumerate(results_tfidf, 1):
    print(f"{i}. [{score:.3f}] {case_text}")

# Example 2: LSA Method
print("\n\n2. LSA Method")
print("-" * 40)
print("(Latent Semantic Analysis)")
print("Best for: Semantic similarity, understanding topics")

results_lsa = tls.case_retrieval(query, cases, method='lsa', n_components=5, top_k=3)

print(f"\nQuery: {query}")
print("\nTop 3 results:")
for i, (case_text, score) in enumerate(results_lsa, 1):
    print(f"{i}. [{score:.3f}] {case_text}")

# Example 3: Word-Average Method
print("\n\n3. Word-Average Method")
print("-" * 40)
print("(Simple averaging of word vectors)")
print("Best for: Fast baseline, interpretable embeddings")

results_word_avg = tls.case_retrieval(query, cases, method='word_avg', top_k=3)

print(f"\nQuery: {query}")
print("\nTop 3 results:")
for i, (case_text, score) in enumerate(results_word_avg, 1):
    print(f"{i}. [{score:.3f}] {case_text}")

# Example 4: Side-by-Side Comparison
print("\n\n4. Side-by-Side Comparison")
print("-" * 40)

test_queries = [
    "What testing approaches exist?",
    "How to track code changes?",
    "Ways to improve software speed?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    print()

    # Get top result from each method
    top_tfidf = tls.case_retrieval(query, cases, method='tfidf', top_k=1)[0]
    top_lsa = tls.case_retrieval(query, cases, method='lsa', top_k=1)[0]
    top_word_avg = tls.case_retrieval(query, cases, method='word_avg', top_k=1)[0]

    print(f"  TF-IDF:    [{top_tfidf[1]:.3f}] {top_tfidf[0][:50]}...")
    print(f"  LSA:       [{top_lsa[1]:.3f}] {top_lsa[0][:50]}...")
    print(f"  Word-Avg:  [{top_word_avg[1]:.3f}] {top_word_avg[0][:50]}...")

# Example 5: Method Selection Guide
print("\n\n5. Method Selection Guide")
print("-" * 40)
print("""
When to use each method:

TF-IDF:
  ✓ Keyword-based search
  ✓ Document retrieval
  ✓ Exact term matching important
  ✓ Large corpus with varied vocabulary
  ✗ Poor at semantic understanding

LSA:
  ✓ Semantic similarity
  ✓ Topic modeling
  ✓ Finding conceptually related documents
  ✓ Reducing dimensionality
  ✗ More computationally expensive

Word-Average:
  ✓ Fast baseline
  ✓ Interpretable results
  ✓ Simple implementation
  ✓ Good for small vocabularies
  ✗ Loses word order information
  ✗ All words weighted equally

Default recommendation: LSA for most reasoning tasks
""")

# Example 6: Temperature + Method Interaction
print("\n6. Temperature + Method Interaction")
print("-" * 40)

query = "How to verify software correctness?"

print(f"Query: {query}\n")

for method in ['tfidf', 'lsa']:
    print(f"\nMethod: {method.upper()}")

    # Use analogical_reasoning to apply temperature
    results_low = tls.analogical_reasoning(query, cases, method=method, temperature=0.5, top_k=2)
    results_high = tls.analogical_reasoning(query, cases, method=method, temperature=2.0, top_k=2)

    print(f"  T=0.5: {results_low[0][2][:40]}... [{results_low[0][1]:.3f}]")
    print(f"  T=2.0: {results_high[0][2][:40]}... [{results_high[0][1]:.3f}]")

print("\n" + "=" * 60)
print("All embedding method examples completed!")
print("=" * 60)
