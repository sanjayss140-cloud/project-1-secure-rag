import pytest
from retriever import HybridRetriever
from models import DocumentChunk

@pytest.fixture
def retriever():
    r = HybridRetriever()
    # Sample synthetic chunks
    test_chunks = [
        DocumentChunk(
            chunk_id="doc1#chunk_0",
            document_id="doc1",
            chunk_index=0,
            content="PostgreSQL database replication uses write-ahead logging (WAL) for streaming data.",
            token_count=10
        ),
        DocumentChunk(
            chunk_id="doc1#chunk_1",
            document_id="doc1",
            chunk_index=1,
            content="Apache Kafka coordinates event streams with partitioned distributed commit logs.",
            token_count=10
        ),
        DocumentChunk(
            chunk_id="doc2#chunk_0",
            document_id="doc2",
            chunk_index=0,
            content="Machine learning pipelines deploy HuggingFace transformers for text embeddings.",
            token_count=9
        )
    ]
    r.build_indexes(test_chunks)
    return r

def test_dense_search(retriever):
    results = retriever.dense_search("database replication and WAL logs", top_k=2)
    assert len(results) > 0
    top_idx, score = results[0]
    # The first document about PostgreSQL should rank highest
    assert top_idx == 0
    assert isinstance(score, float)

def test_sparse_bm25_search(retriever):
    results = retriever.sparse_search("Kafka event streams", top_k=2)
    assert len(results) > 0
    top_idx, score = results[0]
    # The chunk with Kafka should rank first
    assert top_idx == 1
    assert score > 0.0

def test_hybrid_rrf_scoring(retriever):
    results = retriever.hybrid_search("PostgreSQL streaming", top_k=2)
    assert len(results) > 0
    first_result = results[0]
    assert first_result.rrf_score > 0.0
    assert first_result.chunk.document_id == "doc1"
