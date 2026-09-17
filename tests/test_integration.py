import pytest
from retriever import HybridRetriever
from ingestion import DocumentIngestionPipeline

def test_end_to_end_privacy_and_retrieval():
    raw_doc = (
        "Internal Incident Summary:\n"
        "Lead Responder: Dr. Sanjay (sanjay@corp.com, phone: 555-987-6543).\n"
        "Core cluster node at 192.168.10.5 experienced high packet loss.\n"
        "Remediation involved switching cluster traffic to hot-standby node at 10.0.1.25."
    )

    retriever = HybridRetriever()
    pipeline = DocumentIngestionPipeline(retriever)

    # Ingest and index
    chunks, audit_rec = pipeline.process_and_index_document(raw_doc, filename="incident_report.txt")

    # Assert privacy boundary
    assert audit_rec.detected_entities_count >= 3
    for chunk in chunks:
        assert "sanjay.kumar@corp.com" not in chunk.content
        assert "192.168.10.5" not in chunk.content

    # Query hybrid engine
    query = "What was the node that experienced packet loss?"
    scored_results = retriever.hybrid_search(query, top_k=2)

    assert len(scored_results) > 0
    top_chunk = scored_results[0].chunk
    assert "packet loss" in top_chunk.content.lower()
    # Confirm it has citation metadata
    assert top_chunk.document_id == "incident_report.txt"
    assert "incident_report.txt#chunk_" in top_chunk.chunk_id
