#!/usr/bin/env python3

import tidyllm_sentence as tls

# Test sentences that demonstrate preprocessing benefits
sentences = [
    "The cats are running quickly through the park",
    "A cat runs fast in the garden", 
    "Dogs love playing fetch games outdoors",
    "The dog is playing with a ball outside",
    "Machine learning algorithms are fascinating tools",
    "AI and ML techniques help solve complex problems"
]

print("=== Preprocessing Impact Demo ===\n")

print("Original sentences:")
for i, sent in enumerate(sentences):
    print(f"  {i+1}. {sent}")
print()

# Test 1: Without preprocessing (old way)
print("1. WITHOUT Preprocessing (minimal pipeline):")
old_embeddings, old_model = tls.tfidf_fit_transform(sentences, preprocessor=tls.MINIMAL_PIPELINE)
print(f"   Vocabulary size: {old_model.vocab_size}")
print(f"   Sample vocabulary: {list(old_model.vocabulary.keys())[:10]}")

# Find similarity between "cats running" and "cat runs"
old_sim = tls.cosine_similarity(old_embeddings[0], old_embeddings[1])
print(f"   Similarity 'cats running' vs 'cat runs': {old_sim:.4f}")
print()

# Test 2: With preprocessing (new way)  
print("2. WITH Preprocessing (standard pipeline):")
new_embeddings, new_model = tls.tfidf_fit_transform(sentences, preprocessor=tls.STANDARD_PIPELINE)
print(f"   Vocabulary size: {new_model.vocab_size}")
print(f"   Sample vocabulary: {list(new_model.vocabulary.keys())[:10]}")

# Find similarity between same sentences
new_sim = tls.cosine_similarity(new_embeddings[0], new_embeddings[1])
print(f"   Similarity 'cats running' vs 'cat runs': {new_sim:.4f}")
print(f"   IMPROVEMENT: {new_sim - old_sim:+.4f}")
print()

# Test 3: Show preprocessing pipeline
print("3. Preprocessing Pipeline Demo:")
test_sentence = "The cats are running quickly through the parks!"

print(f"   Original: '{test_sentence}'")

minimal_tokens = tls.MINIMAL_PIPELINE.process(test_sentence)
print(f"   Minimal:  {minimal_tokens}")

standard_tokens = tls.STANDARD_PIPELINE.process(test_sentence)  
print(f"   Standard: {standard_tokens}")

aggressive_tokens = tls.AGGRESSIVE_PIPELINE.process(test_sentence)
print(f"   Aggressive: {aggressive_tokens}")
print()

# Test 4: Stemming demonstration
print("4. Stemming Examples:")
test_words = ['running', 'runs', 'cats', 'quickly', 'playing', 'games', 'fascinated', 'algorithms']

print("   Word      -> Simple Stem -> Porter Stem")
for word in test_words:
    simple = tls.simple_stem(word)
    porter = tls.porter_stem(word) 
    print(f"   {word:10} -> {simple:10} -> {porter}")
print()

# Test 5: Stop word filtering
print("5. Stop Word Removal:")
test_text = "The quick brown fox jumps over the lazy dog"
tokens_before = tls.word_tokenize(test_text)
tokens_after = tls.remove_stopwords(tokens_before)

print(f"   Before: {tokens_before}")
print(f"   After:  {tokens_after}")
print(f"   Removed: {len(tokens_before) - len(tokens_after)} stop words")
print()

# Test 6: Semantic search improvement
print("6. Semantic Search Improvement:")
query = "fast cats"

# Without preprocessing - use same model
old_query_emb = old_model._sentence_to_tfidf(query)
old_results = tls.semantic_search(old_query_emb, old_embeddings, top_k=2)

# With preprocessing - use same model
new_query_emb = new_model._sentence_to_tfidf(query)
new_results = tls.semantic_search(new_query_emb, new_embeddings, top_k=2)

print(f"   Query: '{query}'")
print(f"   Without preprocessing - Best match: '{sentences[old_results[0][0]]}' (score: {old_results[0][1]:.4f})")
print(f"   With preprocessing    - Best match: '{sentences[new_results[0][0]]}' (score: {new_results[0][1]:.4f})")
print(f"   Score improvement: {new_results[0][1] - old_results[0][1]:+.4f}")

print("\n=== Preprocessing Dramatically Improves Embeddings! ===")
print("Key benefits:")
print("• Stemming connects related words (run/running/runs)")  
print("• Stop word removal focuses on content")
print("• Better similarity scores between related sentences")
print("• Smaller, more focused vocabularies")
print("• Improved semantic search results")