# System Architecture: Secure & Compliant RAG Platform

## Overview
The Secure & Compliant RAG Platform provides an enterprise-grade document search and question-answering architecture that strictly preserves data confidentiality and implements zero-retention external communication policies.

## Architectural Layers

1. **Ingestion & Data Minimization Gateway (`ingestion.py`, `audit.py`)**
   - Intercepts unstructured input documents (`.txt`, `.md`).
   - Identifies sensitive entities (Names, Emails, Phone Numbers, IP Addresses, Financial Identifiers).
   - Generates deterministic redaction tokens (e.g. `<EMAIL_ADDRESS_1>`, `<IP_ADDRESS_1>`).
   - Records immutable audit events in `storage/audit_log.jsonl`.
   - Generates recursive character chunks with bounded size and overlap.

2. **Dual-Retrieval Hybrid Store (`retriever.py`)**
   - **Dense Embedding Layer:** Computes 384-dimensional vector embeddings via `sentence-transformers/all-MiniLM-L6-v2`. Normalized embeddings are indexed into a FAISS Inner-Product (cosine similarity) index.
   - **Sparse Lexical Layer:** Builds an inverted index using BM25Okapi for keyword matching.
   - **Reciprocal Rank Fusion (RRF):** Fuses dense and sparse rankings using the standard reciprocal rank formula:
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)} \quad (k = 60)$$

3. **Grounded Generation & Synthesis (`app.py`)**
   - Formulates prompts with strict context boundaries.
   - Dispatches inference requests exclusively to the local Ollama daemon (`qwen2.5:3b`).
   - Returns answers alongside verifiable chunk IDs, ranks, and citation metadata.
