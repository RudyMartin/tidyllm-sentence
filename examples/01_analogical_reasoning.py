"""
Example 1: Analogical Reasoning with Temperature Control

Demonstrates case-based reasoning from tidyllm_sentence.reasoning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tidyllm_sentence as tls

print("=" * 60)
print("TidyLLM-Sentence: Analogical Reasoning Examples")
print("=" * 60)

# Example 1: Basic Case Retrieval
print("\n1. Basic Case Retrieval")
print("-" * 40)

cases = [
    "Data validation is required for MVS compliance",
    "Schema checks must be performed before processing",
    "Code review ensures quality and security",
    "Unit tests verify individual components",
    "Integration tests check system interactions"
]

query = "How do we ensure data quality?"

results = tls.case_retrieval(query, cases, method='lsa', top_k=3)

print(f"Query: {query}\n")
print("Top 3 relevant cases:")
for i, (case_text, score) in enumerate(results, 1):
    print(f"\n{i}. Similarity: {score:.3f}")
    print(f"   {case_text}")

# Example 2: Analogical Reasoning with Temperature
print("\n\n2. Temperature-Controlled Analogical Reasoning")
print("-" * 40)

query = "What validation steps are needed?"

print(f"Query: {query}\n")

# T=0: Only exact matches
print("T=0.0 (Symbolic - exact matches only):")
results_symbolic = tls.analogical_reasoning(query, cases, top_k=3, temperature=0.0)
print(f"  Found {len(results_symbolic)} exact matches")
for idx, score, text in results_symbolic[:2]:
    print(f"  - [{score:.3f}] {text[:50]}...")

# T=1.0: Standard ranking
print("\nT=1.0 (Standard ranking):")
results_standard = tls.analogical_reasoning(query, cases, top_k=3, temperature=1.0)
for idx, score, text in results_standard:
    print(f"  - [{score:.3f}] {text[:50]}...")

# T=2.0: More diverse
print("\nT=2.0 (More exploratory):")
results_exploratory = tls.analogical_reasoning(query, cases, top_k=3, temperature=2.0)
for idx, score, text in results_exploratory:
    print(f"  - [{score:.3f}] {text[:50]}...")

# Example 3: Similarity-Based Inference
print("\n\n3. Similarity-Based Inference")
print("-" * 40)

knowledge_base = [
    "The sky is blue due to Rayleigh scattering",
    "Grass is green because of chlorophyll",
    "Water appears blue due to selective absorption",
    "Sunsets are red because of atmospheric scattering"
]

query = "Why is the sky blue?"

result = tls.similarity_based_inference(query, knowledge_base, threshold=0.3, method='lsa')

print(f"Query: {query}")
print(f"\nBest match: {result['best_match'][0] if result['best_match'] else 'None'}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Total matches above threshold: {result['num_matches']}")

# Example 4: Temperature Sweep
print("\n\n4. Temperature Sweep Analysis")
print("-" * 40)

cases = [
    "Requirement A must be validated",
    "Requirement B needs approval",
    "Requirement C should be documented"
]

query = "What requirements need validation?"

sweep_results = tls.temperature_sweep(
    query, cases,
    temperatures=[0.0, 0.5, 1.0, 2.0],
    method='lsa',
    top_k=2
)

print(f"Query: {query}\n")
for temp, results in sweep_results.items():
    print(f"Temperature {temp}:")
    for idx, score, text in results:
        print(f"  [{score:.3f}] {text}")
    print()

# Example 5: Multi-Query Reasoning
print("\n5. Multi-Query Reasoning")
print("-" * 40)

cases = [
    "Data validation checks input quality",
    "Schema validation ensures structure",
    "Business rule validation enforces logic",
    "Security validation prevents threats"
]

queries = [
    "How to validate data?",
    "What checks are needed?",
    "How to ensure quality?"
]

result = tls.multi_query_reasoning(
    queries, cases,
    method='lsa',
    top_k=2,
    aggregation='voting'  # Average scores across queries
)

print("Queries:")
for q in queries:
    print(f"  - {q}")

print("\nAggregated results (voting):")
for case_text, avg_score in result['aggregated'][:3]:
    print(f"  [{avg_score:.3f}] {case_text}")

print("\n" + "=" * 60)
print("All analogical reasoning examples completed!")
print("=" * 60)
