#!/usr/bin/env python3

import sys
sys.path.insert(0, '../tlm')  # Use local version first

import tidyllm_sentence as tls
import tlm

# Simple test
sentences = ["The cat sat on the mat", "Dogs love to play"]

# Test TF-IDF embeddings
embeddings, model = tls.tfidf_fit_transform(sentences)
print(f"Generated embeddings: {len(embeddings)}")
print(f"Embedding shape: {len(embeddings[0])}")

# Test tlm normalization
try:
    normalized_embeddings = tlm.l2_normalize(embeddings)
    print("L2 normalization successful")
    print(f"First embedding norm: {tlm.norm(normalized_embeddings[0]):.4f}")
except Exception as e:
    print(f"L2 normalization error: {e}")
    import traceback
    traceback.print_exc()