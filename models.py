from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class PIIEntity(BaseModel):
    entity_type: str = Field(description="Category of PII detected, e.g., EMAIL_ADDRESS, PERSON, PHONE_NUMBER, IP_ADDRESS")
    start: int = Field(description="Start character index in the source text")
    end: int = Field(description="End character index in the source text")
    score: float = Field(description="Detection confidence score between 0.0 and 1.0")
    replacement_token: str = Field(description="Deterministic anonymization placeholder token")
    original_value_masked: str = Field(description="Partially masked representation for debugging (e.g. j***@corp.com)")

class AuditRecord(BaseModel):
    record_id: str = Field(description="Unique UUID for this audit event")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source_document: str = Field(description="Filename or source document identifier")
    detected_entities_count: int = Field(description="Total PII items detected")
    entities: List[PIIEntity] = Field(default_factory=list, description="List of detected PII entities and their redactions")
    status: str = Field(default="PROCESSED", description="Status of redaction and sanitization")

class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Unique identifier formatted as doc_name#chunk_index")
    document_id: str = Field(description="Source document name or identifier")
    chunk_index: int = Field(description="0-indexed position within the document")
    content: str = Field(description="Sanitized and anonymized textual content")
    token_count: int = Field(description="Approximate word/token count in chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attributes")

class ScoredChunk(BaseModel):
    chunk: DocumentChunk = Field(description="The underlying document chunk")
    dense_rank: Optional[int] = Field(default=None, description="Rank from dense FAISS search")
    sparse_rank: Optional[int] = Field(default=None, description="Rank from sparse BM25 search")
    rrf_score: float = Field(description="Reciprocal Rank Fusion score")

class QueryResponse(BaseModel):
    query: str = Field(description="Original user query")
    sanitized_query: str = Field(description="Query after PII scrubbing")
    query_pii_detected: int = Field(description="Number of PII entities found in query")
    answer: str = Field(description="Generated answer from the local LLM")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Explicit chunk citations used for grounding")
    retrieval_latency_ms: float = Field(description="Time taken for dense + sparse + RRF retrieval in milliseconds")
    generation_latency_ms: float = Field(description="Time taken for local LLM inference in milliseconds")
    model_name: str = Field(description="Name of the local LLM used")
