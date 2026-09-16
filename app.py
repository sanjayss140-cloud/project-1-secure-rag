import time
import requests
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from config import settings
from models import DocumentChunk, ScoredChunk, QueryResponse
from audit import pii_engine
from retriever import HybridRetriever
from ingestion import DocumentIngestionPipeline

# Set Page Config
st.set_page_config(
    page_title="Secure & Compliant RAG Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize singletons in session state
@st.cache_resource
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever()

retriever = get_hybrid_retriever()
ingestion_pipeline = DocumentIngestionPipeline(retriever)

def check_ollama_health() -> Dict[str, Any]:
    """Verify Ollama server and model availability."""
    try:
        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3.0)
        if resp.status_code == 200:
            models_data = resp.json().get("models", [])
            model_names = [m.get("name") for m in models_data]
            has_target = any(settings.OLLAMA_MODEL in name for name in model_names)
            return {"status": "ONLINE", "available": True, "models": model_names, "has_target": has_target}
    except Exception as e:
        return {"status": "OFFLINE", "available": False, "error": str(e)}
    return {"status": "UNKNOWN", "available": False}

def query_ollama(prompt: str, system_prompt: str) -> str:
    """Invokes local Ollama inference via REST API with strict timeouts."""
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9
        }
    }
    try:
        resp = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=settings.OLLAMA_REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            return f"Ollama Error (HTTP {resp.status_code}): {resp.text}"
    except requests.exceptions.Timeout:
        return "Error: Ollama generation timed out. Please check local machine CPU/GPU load."
    except Exception as e:
        return f"Inference Error: {str(e)}"

# --- SIDEBAR: SYSTEM HEALTH & INGESTION ---
with st.sidebar:
    st.title("🛡️ Governance & Ops")
    st.caption("Privacy-First Hybrid RAG Platform")
    st.markdown("---")

    # Health Check
    health = check_ollama_health()
    if health.get("status") == "ONLINE":
        st.success(f"🟢 Ollama Online ({settings.OLLAMA_MODEL})")
        if not health.get("has_target"):
            st.warning(f"Target model '{settings.OLLAMA_MODEL}' not found. Available: {health.get('models')}")
    else:
        st.error("🔴 Ollama Server Offline")
        st.caption("Run: `ollama run qwen2.5:3b` in a terminal.")

    st.markdown("---")
    st.subheader("📚 Index Statistics")
    indexed_chunks_count = len(retriever.chunks)
    unique_docs = len(set(c.document_id for c in retriever.chunks))
    col_stat1, col_stat2 = st.columns(2)
    col_stat1.metric("Documents", unique_docs)
    col_stat2.metric("Chunks", indexed_chunks_count)

    st.markdown("---")
    st.subheader("📥 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Text / Markdown Document", type=["txt", "md"])
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8", errors="replace")
        if st.button("Scrub PII & Index Document", use_container_width=True):
            with st.spinner("Analyzing PII & building hybrid index..."):
                chunks, audit_rec = ingestion_pipeline.process_and_index_document(file_text, uploaded_file.name)
                st.success(f"Indexed {len(chunks)} sanitized chunks!")
                st.info(f"Redacted {audit_rec.detected_entities_count} sensitive PII entities.")

    st.markdown("---")
    # Quick Load Default Sample
    if st.button("Load Built-in Sample Report", use_container_width=True):
        sample_path = settings.DATA_DIR / "sample_confidential.txt"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                sample_text = f.read()
            with st.spinner("Processing built-in confidential sample..."):
                chunks, audit_rec = ingestion_pipeline.process_and_index_document(sample_text, "sample_confidential.txt")
                st.success(f"Loaded sample! ({len(chunks)} chunks, {audit_rec.detected_entities_count} PII items redacted)")
                st.rerun()

# --- MAIN PAGE ---
st.title("🔒 Privacy-Preserving Enterprise RAG")
st.markdown("""
This platform executes **PII redaction**, **Hybrid Retrieval (Dense FAISS + Sparse BM25)**, 
and **Reciprocal Rank Fusion (RRF)**. All inference runs strictly on a **local LLM** without cloud leakage.
""")

tab_chat, tab_audit, tab_chunks = st.tabs(["💬 Query & Grounded Chat", "📜 Privacy Audit Ledger", "🧩 Chunks Inspector"])

# --- TAB 1: CHAT & RETRIEVAL ---
with tab_chat:
    user_query = st.text_input("Enter your question regarding enterprise documents:", placeholder="e.g. What are the IP addresses and failover setups for Project Titan?")

    if st.button("Search & Generate Answer", type="primary", use_container_width=False):
        if not user_query.strip():
            st.warning("Please enter a question.")
        elif not retriever.chunks:
            st.warning("No documents indexed yet. Please upload a document or click 'Load Built-in Sample Report' in the sidebar.")
        else:
            t0 = time.time()
            # 1. Scrub PII from query itself
            sanitized_query, query_audit = pii_engine.scan_and_anonymize(user_query, source_doc="user_query")
            
            # 2. Hybrid Retrieval with RRF
            scored_chunks = retriever.hybrid_search(sanitized_query, top_k=settings.FINAL_TOP_K)
            retrieval_ms = (time.time() - t0) * 1000

            if not scored_chunks:
                st.warning("No relevant information found in indexed documents.")
            else:
                # 3. Assemble Grounded Context
                context_blocks = []
                citations = []
                for sc in scored_chunks:
                    c = sc.chunk
                    context_blocks.append(f"[{c.chunk_id}]:\n{c.content}")
                    citations.append({
                        "chunk_id": c.chunk_id,
                        "document": c.document_id,
                        "rrf_score": sc.rrf_score,
                        "dense_rank": sc.dense_rank,
                        "sparse_rank": sc.sparse_rank
                    })
                
                context_str = "\n\n".join(context_blocks)

                system_prompt = (
                    "You are an enterprise AI assistant with strict compliance and privacy rules.\n"
                    "Instructions:\n"
                    "1. Answer the question SOLELY using the provided context chunks.\n"
                    "2. Explicitly cite the chunk identifiers (e.g. [sample_confidential.txt#chunk_0]) for every factual claim.\n"
                    "3. If the answer is not present in the context, clearly state: 'The provided documents do not contain information to answer this question.'\n"
                    "4. Never hallucinate or assume unstated credentials."
                )

                augmented_prompt = f"CONTEXT INFORMATION:\n{context_str}\n\nQUESTION:\n{sanitized_query}\n\nGROUNDED ANSWER WITH CITATIONS:"

                # 4. Local Inference
                t_gen0 = time.time()
                with st.spinner("Generating answer locally via Ollama..."):
                    llm_answer = query_ollama(augmented_prompt, system_prompt)
                generation_ms = (time.time() - t_gen0) * 1000

                # Display Results
                st.subheader("💡 Grounded Answer")
                st.markdown(llm_answer)

                st.markdown("---")
                # Observability & Metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Retrieval Latency", f"{retrieval_ms:.1f} ms")
                col_m2.metric("Inference Latency", f"{generation_ms:.1f} ms")
                col_m3.metric("Chunks Retrieved", len(scored_chunks))
                col_m4.metric("Query PII Redacted", query_audit.detected_entities_count)

                # Explicit Citations Table
                st.subheader("📌 Verifiable Citations & Rank Fusion Details")
                citation_df = pd.DataFrame(citations)
                st.dataframe(citation_df, use_container_width=True)

                # Context expander
                with st.expander("🔍 Inspect Grounding Context Chunks"):
                    for sc in scored_chunks:
                        st.markdown(f"**Chunk ID:** `{sc.chunk.chunk_id}` | **RRF Score:** `{sc.rrf_score}` | Dense Rank: `{sc.dense_rank}` | Sparse Rank: `{sc.sparse_rank}`")
                        st.text(sc.chunk.content)
                        st.divider()

# --- TAB 2: AUDIT LEDGER ---
with tab_audit:
    st.subheader("📜 Immutable Privacy Audit Ledger")
    st.caption("Tracks all PII redactions, entity classifications, and redaction tokens before indexing.")
    
    logs = pii_engine.read_audit_logs(limit=30)
    if not logs:
        st.info("No audit entries recorded yet. Ingest documents to generate audit trails.")
    else:
        flat_records = []
        for l in logs:
            for entity in l.get("entities", []):
                flat_records.append({
                    "Timestamp": l.get("timestamp"),
                    "Source Document": l.get("source_document"),
                    "Entity Type": entity.get("entity_type"),
                    "Placeholder": entity.get("replacement_token"),
                    "Masked Preview": entity.get("original_value_masked"),
                    "Confidence": entity.get("score")
                })
        if flat_records:
            st.dataframe(pd.DataFrame(flat_records), use_container_width=True)
        else:
            st.info("Documents processed with 0 sensitive PII detections.")

# --- TAB 3: CHUNKS INSPECTOR ---
with tab_chunks:
    st.subheader("🧩 Indexed Chunks (Sanitized)")
    if not retriever.chunks:
        st.info("No chunks currently in memory.")
    else:
        chunks_data = [
            {
                "Chunk ID": c.chunk_id,
                "Document": c.document_id,
                "Tokens": c.token_count,
                "Content Preview": c.content[:120] + "..." if len(c.content) > 120 else c.content
            }
            for c in retriever.chunks
        ]
        st.dataframe(pd.DataFrame(chunks_data), use_container_width=True)
