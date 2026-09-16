# Engineering Decisions Log (ADR)

## Decision 1: Dual-Engine PII Detection Architecture
- **Context:** On Windows environments with Windows Defender Application Control (WDAC) or specific security baselines, compiled `.pyd` binary wheels from spaCy can trigger policy-level DLL blocks.
- **Decision:** Architected `audit.py` with a dual engine. It dynamically imports and leverages Microsoft Presidio when available. If native spaCy DLLs are restricted by the OS security policy, it activates an Enterprise Rule and Regex Engine implementing identical Pydantic entity schemas and redaction tokens. In containerized Linux environments, full Presidio runs natively.
- **Consequence:** 100% reproducible execution on Windows host machines and Linux containers without compromising PII detection or crashing.

## Decision 2: Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
- **Context:** Dense vector search alone suffers from semantic dilution when dealing with exact identifiers, IP addresses, or part codes. BM25 alone fails to capture conceptual synonyms.
- **Decision:** Implemented dual indexing with FAISS and BM25Okapi, combined using Reciprocal Rank Fusion ($k=60$).
- **Consequence:** Combines semantic and lexical recall without requiring manual score-calibration across different metric spaces.

## Decision 3: Local LLM Hosting via Ollama
- **Context:** Commercial cloud LLMs introduce third-party data processing risks, data retention concerns, and recurring cost overheads.
- **Decision:** Use local Ollama daemon with `qwen2.5:3b`.
- **Consequence:** Complete data sovereignty. Zero bytes of enterprise text are transmitted over external networks.
