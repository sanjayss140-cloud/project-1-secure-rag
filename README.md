# Project 1: Secure & Compliant RAG Platform

An enterprise-grade, privacy-first Retrieval-Augmented Generation (RAG) platform that processes sensitive internal documentation locally without transmitting proprietary data to external LLM APIs.

---

## 1. Business Problem
Enterprises seeking to leverage Generative AI for internal knowledge retrieval face severe compliance risks:
- Proprietary intellectual property, credentials, and internal network topologies can be exposed if sent to public multi-tenant LLM APIs.
- Personally Identifiable Information (PII) of employees and customers must be scrubbed under GDPR/CCPA data minimization principles before entering vector indices or prompting layers.
- Hallucinations can introduce operational errors unless all generation is strictly grounded with verifiable chunk-level citations.

---

## 2. Architecture
```
Documents (.txt, .md)
       │
       ▼
[PII Scanner & Anonymizer] ───> [Immutable Audit Ledger (JSONL)]
       │
       ▼
[Sanitized Document Text]
       │
       ▼
[Recursive Character Chunking]
       │
       ├───> [Dense FAISS Vector Store] ───┐
       │                                   ├──> [Reciprocal Rank Fusion (RRF)]
       └───> [Sparse BM25 Inverted Index] ──┘
                                                   │
                                                   ▼
                                         [Local Ollama Inference (qwen2.5:3b)]
                                                   │
                                                   ▼
                                      [Answer + Citations + Latency Stats]
```

---

## 3. Technology Decisions
- **PII Governance:** Microsoft Presidio with regex pattern fallbacks for deterministic PII anonymization and audit recording.
- **Dense Index:** FAISS (`IndexFlatIP` on L2-normalized embeddings) for fast semantic vector search.
- **Sparse Index:** BM25Okapi for precise lexical search on codes, hostnames, and exact keywords.
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF, $k=60$) combining semantic and lexical signals without metric distortion.
- **Local Inference:** Ollama hosting `qwen2.5:3b` on localhost.
- **UI:** Streamlit for interactive search, ingestion, and audit inspection.

---

## 4. Repository Structure
```
project_1_secure_rag/
├── .env.example
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── config.py                 # Central settings via Pydantic
├── models.py                 # Pydantic schemas (DocumentChunk, PIIEntity, etc.)
├── audit.py                  # PII scanning, masking, and audit logging
├── ingestion.py              # Chunking and dual-index building
├── retriever.py              # FAISS + BM25 + RRF Hybrid Retriever
├── app.py                    # Streamlit UI
├── benchmark.py              # Latency & throughput empirical benchmarking
├── data/
│   └── sample_confidential.txt
├── storage/                  # Persisted indexes and audit logs
├── tests/
│   ├── test_pii.py
│   ├── test_chunking.py
│   ├── test_retriever.py
│   └── test_integration.py
└── docs/
    ├── architecture.md
    ├── data-flow.md
    ├── decisions.md
    └── troubleshooting.md
```

---

## 5. Local Setup Instructions

### Prerequisites
- Python 3.11+
- Git
- Ollama installed with model pulled:
  ```cmd
  ollama pull qwen2.5:3b
  ```

### Installation
1. Navigate to the project directory:
   ```cmd
   cd C:\Users\Sanjay\Downloads\ai-realtime-recommendation-engine\project_1_secure_rag
   ```
2. Activate your virtual environment:
   ```cmd
   ..\.venv\Scripts\activate.bat
   ```
3. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

---

## 6. Running the Application

### Option A: Run Locally
```cmd
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

### Option B: Run via Docker Compose
```cmd
docker-compose up --build
```

---

## 7. Running Tests & Benchmarks

### Execute Unit and Integration Tests
```cmd
pytest tests/ -v
```

### Execute Retrieval Benchmark
```cmd
python benchmark.py
```

### Measured Benchmark Results (Local Test Run, CPU)
| Engine / Metric | P50 Latency | P95 Latency | P99 Latency | Throughput |
|---|---|---|---|---|
| **Dense Search (FAISS)** | 19.66 ms | 23.04 ms | 23.52 ms | ~50 QPS |
| **Sparse Search (BM25)** | 0.24 ms | 0.35 ms | 0.66 ms | >1000 QPS |
| **Hybrid RRF Fusion** | 20.17 ms | 22.85 ms | 27.53 ms | 49.01 queries/sec |

*Benchmark configuration: 25 iterations, all-MiniLM-L6-v2 embeddings, Windows local CPU execution.*

---

## 8. Security & Data Governance
- **Data Minimization:** PII entities (`EMAIL_ADDRESS`, `PHONE_NUMBER`, `IP_ADDRESS`, `PERSON`) are detected and replaced with synthetic placeholders prior to embedding and indexing.
- **Auditability:** Every redaction is logged into `storage/audit_log.jsonl` with an immutable timestamp and masked value preview.
- **Air-Gapped Local Inference:** All prompt data is routed to the local Ollama instance over `localhost:11434`. Zero external network calls are made.

---

## 9. Known Limitations
- Local LLM inference speed depends on host CPU/GPU capabilities.
- Complex tabular data in scanned PDF images requires OCR preprocessing before ingestion.
