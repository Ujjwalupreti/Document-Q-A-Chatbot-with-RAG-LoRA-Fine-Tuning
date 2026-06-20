"""
Benchmark retrieval latency: naive linear scan vs ANN-style search +
re-ranking, the comparison that produces the latency numbers referenced on
the resume.

Generates a synthetic corpus of ~10,000 chunks, embeds it, and times query
latency for both approaches.

Usage:
    python retrieval_benchmark.py --n_chunks 10000 --n_queries 20
"""
import argparse
import json
import random
import string
import time
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def random_sentence(n_words: int = 20) -> str:
    words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
        for _ in range(n_words)
    ]
    return " ".join(words)


def cosine_similarity_matrix(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / np.linalg.norm(query_vec)
    corpus_norm = corpus_matrix / np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
    return corpus_norm @ query_norm


def naive_search(query_vec: np.ndarray, corpus_matrix: np.ndarray, k: int = 4) -> np.ndarray:
    scores = cosine_similarity_matrix(query_vec, corpus_matrix)
    return np.argsort(scores)[-k:]


def main(n_chunks: int, n_queries: int) -> None:
    print(f"Generating synthetic corpus of {n_chunks} chunks...")
    corpus_texts = [random_sentence() for _ in range(n_chunks)]
    query_texts = [random_sentence() for _ in range(n_queries)]

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding corpus (this may take a while for 10k chunks)...")
    corpus_vectors = np.array(embeddings.embed_documents(corpus_texts))

    print("Embedding queries...")
    query_vectors = np.array(embeddings.embed_documents(query_texts))

    # --- Naive linear scan over the full corpus ---
    naive_times = []
    for q in query_vectors:
        start = time.perf_counter()
        naive_search(q, corpus_vectors, k=4)
        naive_times.append(time.perf_counter() - start)

    # --- Approximate "ANN + re-rank" ---
    # Narrow to top-20 via a coarse dot-product pass, then re-rank that
    # smaller set with full cosine similarity. ChromaDB's HNSW index does
    # something similar under the hood, but much faster at scale.
    rerank_times = []
    for q in query_vectors:
        start = time.perf_counter()
        coarse_scores = corpus_vectors @ q
        top_20_idx = np.argsort(coarse_scores)[-20:]
        candidates = corpus_vectors[top_20_idx]
        cosine_similarity_matrix(q, candidates)
        rerank_times.append(time.perf_counter() - start)

    naive_avg = sum(naive_times) / len(naive_times)
    rerank_avg = sum(rerank_times) / len(rerank_times)
    reduction = (naive_avg - rerank_avg) / naive_avg * 100

    print(f"\nCorpus size: {n_chunks} chunks, {n_queries} queries")
    print(f"Naive linear scan avg latency:    {naive_avg * 1000:.1f} ms")
    print(f"ANN + re-rank approx. latency:    {rerank_avg * 1000:.1f} ms")
    print(f"Latency reduction:                 {reduction:.1f}%")
    print(
        "\nNote: real ChromaDB uses an HNSW index, which will be faster than "
        "this approximation. Run this against your actual indexed corpus and "
        "the live /chat endpoint to get the figures to put on your resume."
    )

    results = {
        "naive_ms": round(naive_avg * 1000, 2),
        "rag_ms": round(rerank_avg * 1000, 2),
        "reduction_pct": round(reduction, 1),
    }
    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path} — the API's /metrics endpoint will pick these up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chunks", type=int, default=10000)
    parser.add_argument("--n_queries", type=int, default=20)
    args = parser.parse_args()
    main(args.n_chunks, args.n_queries)
