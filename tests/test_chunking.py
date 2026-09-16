import pytest
from retriever import HybridRetriever
from ingestion import DocumentIngestionPipeline

@pytest.fixture
def ingestion_pipeline():
    retriever = HybridRetriever()
    return DocumentIngestionPipeline(retriever)

def test_chunk_splitting_and_metadata(ingestion_pipeline):
    text = (
        "Enterprise architectural design involves high-availability multi-region deployments. "
        "Each cluster maintains independent consensus state machines. "
        "In the event of network partitions, split-brain mitigation relies on quorum heartbeats. "
        "Failover thresholds are calibrated to minimize failback churn."
    )
    chunks = ingestion_pipeline.chunk_text(text, document_id="doc_arch_01", chunk_size=80, overlap=15)
    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert c.document_id == "doc_arch_01"
        assert c.chunk_index == idx
        assert c.chunk_id == f"doc_arch_01#chunk_{idx}"
        assert len(c.content) > 0
        assert c.token_count > 0

def test_empty_content_chunking(ingestion_pipeline):
    chunks = ingestion_pipeline.chunk_text("", document_id="empty_doc")
    assert len(chunks) == 0

    chunks_ws = ingestion_pipeline.chunk_text("   \n\n   ", document_id="ws_doc")
    assert len(chunks_ws) == 0
