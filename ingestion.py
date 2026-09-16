import os
import logging
from pathlib import Path
from typing import List, Tuple
from config import settings
from models import DocumentChunk, AuditRecord
from audit import pii_engine
from retriever import HybridRetriever

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("document_ingestion")

class DocumentIngestionPipeline:
    """
    Ingestion engine responsible for:
    1. Reading raw files (.txt, .md)
    2. Enforcing the privacy boundary (PII scrubbing before chunking/indexing)
    3. Deterministic chunking with overlap
    4. Dual index building (FAISS + BM25)
    """
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def chunk_text(self, text: str, document_id: str, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP) -> List[DocumentChunk]:
        """
        Recursive character splitter respecting sentence/paragraph boundaries
        while preserving strict chunk indices.
        """
        if not text.strip():
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        text_length = len(text)
        chunk_idx = 0

        while start < text_length:
            end = start + chunk_size
            if end >= text_length:
                chunk_str = text[start:]
            else:
                # Seek nearby newline or space to prevent cutting mid-word
                breakpoint = text.rfind("\n", start, end)
                if breakpoint == -1 or breakpoint <= start:
                    breakpoint = text.rfind(" ", start, end)
                if breakpoint != -1 and breakpoint > start:
                    end = breakpoint
                chunk_str = text[start:end]

            chunk_str = chunk_str.strip()
            if chunk_str:
                chunk_id = f"{document_id}#chunk_{chunk_idx}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_idx,
                    content=chunk_str,
                    token_count=len(chunk_str.split()),
                    metadata={"source": document_id, "char_length": len(chunk_str)}
                ))
                chunk_idx += 1

            start += chunk_size - overlap

        return chunks

    def process_and_index_document(self, content: str, filename: str) -> Tuple[List[DocumentChunk], AuditRecord]:
        """
        Full end-to-end ingestion:
        Raw Text -> PII Scrubbing -> Audit Log -> Chunking -> Hybrid Indexing
        """
        logger.info("Scanning document for PII: %s", filename)
        # 1. Privacy Boundary: Redact sensitive information
        anonymized_text, audit_record = pii_engine.scan_and_anonymize(content, source_doc=filename)
        logger.info("PII scan completed. Detected %d entities.", audit_record.detected_entities_count)

        # 2. Chunking on sanitized text
        chunks = self.chunk_text(anonymized_text, document_id=filename)
        logger.info("Generated %d chunks for document %s", len(chunks), filename)

        # 3. Accumulate with existing chunks and re-index
        existing_chunks = [c for c in self.retriever.chunks if c.document_id != filename]
        all_chunks = existing_chunks + chunks

        # 4. Build FAISS & BM25 indexes
        self.retriever.build_indexes(all_chunks)

        return chunks, audit_record
