#!/usr/bin/env python3

import tidyllm_sentence as tls

# Test data
sentences = [
    "The cat sat on the mat",
    "Dogs love to play fetch", 
    "Machine learning is fascinating",
    "Natural language processing",
    "Cats are independent animals",
    "Dogs are loyal companions"
]

print("=== tidyllm-sentence Demo ===\n")

# Test TF-IDF
print("1. TF-IDF Embeddings:")
tfidf_embs, tfidf_model = tls.tfidf_fit_transform(sentences)
print(f"   Shape: {len(tfidf_embs)}x{len(tfidf_embs[0])}")
print(f"   Vocabulary size: {tfidf_model.vocab_size}")

# Test similarity
query = "cats sitting"
query_emb, _ = tls.tfidf_fit_transform([query])
results = tls.semantic_search(query_emb[0], tfidf_embs, top_k=3)
print(f"   Query: '{query}'")
for rank, (idx, score) in enumerate(results):
    print(f"   {rank+1}. '{sentences[idx]}' (score: {score:.3f})")

print()

# Test Word Averaging
print("2. Word Averaging Embeddings:")
word_embs, word_model = tls.word_avg_fit_transform(sentences, embedding_dim=50, seed=42)
print(f"   Shape: {len(word_embs)}x{len(word_embs[0])}")
print(f"   Uses IDF weighting: {word_model.use_idf}")

print()

# Test N-gram
print("3. Character N-gram Embeddings:")
ngram_embs, ngram_model = tls.ngram_fit_transform(sentences, n=3, ngram_type='char', max_features=100)
print(f"   Shape: {len(ngram_embs)}x{len(ngram_embs[0])}")
print(f"   Sample trigrams: {list(ngram_model.vocabulary.keys())[:5]}")

print()

# Test LSA
print("4. LSA Embeddings:")
lsa_embs, lsa_model = tls.lsa_fit_transform(sentences, n_components=10)
print(f"   Shape: {len(lsa_embs)}x{len(lsa_embs[0])}")
print(f"   Components: {len(lsa_model.components)}")

print()

# Test utilities
print("5. Utilities:")
tokens = tls.word_tokenize("Hello, world! How are you doing?")
print(f"   Tokenization: {tokens}")

char_grams = tls.char_ngrams("hello", n=3)
print(f"   Character 3-grams: {char_grams}")

# Compare embeddings
sim = tls.cosine_similarity(tfidf_embs[0], tfidf_embs[4])  # "cat" vs "cats" 
print(f"   Similarity between cat/cats sentences: {sim:.3f}")

print("\n=== All tests completed! ===")
print("tidyllm-sentence: Transparent, educational sentence embeddings!")