# Data Flow Specification

## 1. Document Ingestion Flow
```
[Raw Document] 
      │
      ▼
[Presidio / Enterprise PII Scanner]
      │
      ├──> [Matches Detected?] ──YES──> [Replace with <TOKEN_N>]
      │                                            │
      │                                            ▼
      │                                   [Append Audit Log JSONL]
      │
      ▼
[Sanitized Document Text]
      │
      ▼
[Recursive Character Splitter (500 chars / 50 overlap)]
      │
      ▼
[Document Chunks + Metadata Binding]
      │
      ├───> [SentenceTransformers MiniLM] ───> [FAISS Index]
      │
      └───> [Whitespace Tokenizer]        ───> [BM25 Inverted Index]
```

## 2. Query & Generation Flow
```
[User Query]
      │
      ▼
[Query PII Scrubbing]
      │
      ├───> [FAISS Top-K Dense Search]  ───┐
      │                                    ├─> [Reciprocal Rank Fusion (k=60)]
      └───> [BM25 Top-K Sparse Search] ───┘
                                                   │
                                                   ▼
                                         [Top-N Ranked Chunks]
                                                   │
                                                   ▼
                                         [Constrained Context Prompt]
                                                   │
                                                   ▼
                                      [Local Ollama (qwen2.5:3b)]
                                                   │
                                                   ▼
                             [Answer + Citations + Latency Metrics]
```
