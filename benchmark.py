import time
import statistics
import numpy as np
from typing import List, Dict
from retriever import HybridRetriever
from ingestion import DocumentIngestionPipeline

def run_retrieval_benchmark(num_iterations: int = 25) -> Dict[str, float]:
    """
    Empirical benchmark measuring Dense, Sparse, and Hybrid (RRF) retrieval latencies.
    Computes real P50, P95, P99 and throughput metrics.
    """
    print("=" * 60)
    print("STARTING EMPIRICAL RAG RETRIEVAL BENCHMARK")
    print(f"Iterations: {num_iterations}")
    print("=" * 60)

    retriever = HybridRetriever()
    pipeline = DocumentIngestionPipeline(retriever)

    # Ingest synthetic corpus
    corpus = [
        "Distributed database replication and write-ahead log mechanisms for durability.",
        "Kafka partition reassignment and consumer group rebalance protocols.",
        "Vector database indexing strategies using HNSW graphs and inverted file lists.",
        "Reciprocal rank fusion algorithms for combining lexical and semantic search results.",
        "Zero-trust security models for microservice token exchange and mTLS encryption."
    ]
    for idx, doc in enumerate(corpus):
        pipeline.process_and_index_document(doc, filename=f"benchmark_doc_{idx}.txt")

    dense_latencies = []
    sparse_latencies = []
    hybrid_latencies = []

    test_queries = [
        "Kafka consumer group rebalance",
        "database replication and WAL logs",
        "vector database indexing and HNSW",
        "reciprocal rank fusion search",
        "microservice token encryption"
    ]

    for i in range(num_iterations):
        q = test_queries[i % len(test_queries)]
        
        # 1. Dense search timing
        t0 = time.perf_counter()
        _ = retriever.dense_search(q, top_k=3)
        dense_latencies.append((time.perf_counter() - t0) * 1000)

        # 2. Sparse search timing
        t0 = time.perf_counter()
        _ = retriever.sparse_search(q, top_k=3)
        sparse_latencies.append((time.perf_counter() - t0) * 1000)

        # 3. Hybrid RRF timing
        t0 = time.perf_counter()
        _ = retriever.hybrid_search(q, top_k=3)
        hybrid_latencies.append((time.perf_counter() - t0) * 1000)

    results = {
        "dense_p50": float(np.percentile(dense_latencies, 50)),
        "dense_p95": float(np.percentile(dense_latencies, 95)),
        "dense_p99": float(np.percentile(dense_latencies, 99)),
        "sparse_p50": float(np.percentile(sparse_latencies, 50)),
        "sparse_p95": float(np.percentile(sparse_latencies, 95)),
        "sparse_p99": float(np.percentile(sparse_latencies, 99)),
        "hybrid_p50": float(np.percentile(hybrid_latencies, 50)),
        "hybrid_p95": float(np.percentile(hybrid_latencies, 95)),
        "hybrid_p99": float(np.percentile(hybrid_latencies, 99)),
        "hybrid_throughput_qps": float(num_iterations / (sum(hybrid_latencies) / 1000.0))
    }

    print("\n--- MEASURED BENCHMARK RESULTS ---")
    print(f"Dense Retrieval (FAISS):  P50={results['dense_p50']:.2f}ms | P95={results['dense_p95']:.2f}ms | P99={results['dense_p99']:.2f}ms")
    print(f"Sparse Retrieval (BM25):  P50={results['sparse_p50']:.2f}ms | P95={results['sparse_p95']:.2f}ms | P99={results['sparse_p99']:.2f}ms")
    print(f"Hybrid Fusion (RRF):      P50={results['hybrid_p50']:.2f}ms | P95={results['hybrid_p95']:.2f}ms | P99={results['hybrid_p99']:.2f}ms")
    print(f"Hybrid Throughput:        {results['hybrid_throughput_qps']:.2f} queries/sec")
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_retrieval_benchmark()
