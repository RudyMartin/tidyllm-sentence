#!/usr/bin/env python3
"""
Test transformer-enhanced embeddings for semantic similarity improvement.
"""

import sys
sys.path.insert(0, '../tlm')

import tidyllm_sentence as tls

# Test sentences with semantic similarity
test_sentences = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "Dogs love to play fetch", 
    "Canines enjoy playing games",
    "Machine learning is powerful",
    "AI techniques solve problems"
]

print("TESTING TRANSFORMER-ENHANCED EMBEDDINGS")
print("=" * 50)

print("Test sentences:")
for i, sentence in enumerate(test_sentences):
    print(f"  {i+1}. {sentence}")

# Test transformer embeddings
print(f"\nGenerating transformer-enhanced embeddings...")
try:
    transformer_embeddings, transformer_model = tls.transformer_fit_transform(
        test_sentences, 
        max_seq_len=16,  # Keep it small for testing
        attention_heads=2
    )
    
    print(f"Success! Generated {len(transformer_embeddings)} embeddings")
    print(f"Embedding dimensions: {len(transformer_embeddings[0])}")
    print(f"Vocabulary size: {transformer_model.vocab_size}")
    
    # Test semantic similarities
    print(f"\nTransformer semantic similarities:")
    
    # Expected high similarities
    pairs_to_test = [
        (0, 1, "cat/mat vs feline/rug"),
        (2, 3, "dogs/fetch vs canines/games"), 
        (4, 5, "ML vs AI"),
    ]
    
    for i, j, description in pairs_to_test:
        sim = tls.cosine_similarity(transformer_embeddings[i], transformer_embeddings[j])
        print(f"  {description}: {sim:.4f}")
    
    # Compare with standard TF-IDF
    print(f"\nComparing with standard TF-IDF:")
    standard_embeddings, standard_model = tls.tfidf_fit_transform(test_sentences)
    
    print(f"Standard TF-IDF similarities:")
    for i, j, description in pairs_to_test:
        sim = tls.cosine_similarity(standard_embeddings[i], standard_embeddings[j])
        print(f"  {description}: {sim:.4f}")
    
    # Calculate improvement
    print(f"\nImprovement analysis:")
    total_improvement = 0
    for i, j, description in pairs_to_test:
        transformer_sim = tls.cosine_similarity(transformer_embeddings[i], transformer_embeddings[j])
        standard_sim = tls.cosine_similarity(standard_embeddings[i], standard_embeddings[j])
        improvement = transformer_sim - standard_sim
        total_improvement += improvement
        print(f"  {description}: {improvement:+.4f}")
    
    print(f"\nAverage improvement: {total_improvement / len(pairs_to_test):+.4f}")
    
    if total_improvement > 0:
        print(f"SUCCESS: Transformer attention improved semantic similarity!")
    else:
        print(f"Hmm: Need to tune the attention mechanism...")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()