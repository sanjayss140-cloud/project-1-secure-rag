import os
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from config import settings
from models import DocumentChunk, ScoredChunk

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("hybrid_retriever")

class HybridRetriever:
    """
    Production Hybrid Retriever combining:
    1. Dense Vector Search (FAISS index with L2 normalized cosine similarity)
    2. Sparse Lexical Search (Rank-BM25 Okapi)
    3. Rank Fusion (Reciprocal Rank Fusion - RRF)
    """
    def __init__(self):
        logger.info("Initializing SentenceTransformer: %s", settings.EMBEDDING_MODEL_NAME)
        try:
            self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, local_files_only=True)
        except Exception:
            self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.faiss_index: Optional[faiss.Index] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        
        # Load existing index if present
        self.load_indexes()

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace & lowercase tokenizer for BM25."""
        return [word.lower() for word in text.split()]

    def build_indexes(self, chunks: List[DocumentChunk]):
        """Builds both FAISS and BM25 indexes from a list of sanitized chunks."""
        if not chunks:
            logger.warning("No chunks provided to build indexes.")
            return

        self.chunks = chunks
        logger.info("Building hybrid index for %d chunks...", len(chunks))

        # 1. Build Dense FAISS Index
        texts = [chunk.content for chunk in chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension) # Inner Product on normalized vectors = Cosine Similarity
        self.faiss_index.add(embeddings.astype(np.float32))

        # 2. Build Sparse BM25 Index
        self.tokenized_corpus = [self.tokenize(t) for t in texts]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)

        # 3. Persist to disk
        self.save_indexes()
        logger.info("Successfully built and persisted FAISS + BM25 indexes.")

    def save_indexes(self):
        """Persists FAISS index and BM25 metadata to storage directory."""
        try:
            settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            # Save FAISS
            faiss_file = settings.STORAGE_DIR / "faiss_index.bin"
            if self.faiss_index:
                faiss.write_index(self.faiss_index, str(faiss_file))

            # Save BM25 and Chunks
            with open(settings.BM25_INDEX_PATH, "wb") as f:
                pickle.dump({
                    "chunks": self.chunks,
                    "bm25": self.bm25_index,
                    "tokenized_corpus": self.tokenized_corpus
                }, f)
            logger.info("Saved index files to %s", settings.STORAGE_DIR)
        except Exception as e:
            logger.error("Error saving indexes: %s", e)

    def load_indexes(self) -> bool:
        """Loads FAISS index and BM25 index from disk if available."""
        faiss_file = settings.STORAGE_DIR / "faiss_index.bin"
        if not faiss_file.exists() or not settings.BM25_INDEX_PATH.exists():
            logger.info("No pre-existing indexes found on disk.")
            return False

        try:
            self.faiss_index = faiss.read_index(str(faiss_file))
            with open(settings.BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.bm25_index = data["bm25"]
                self.tokenized_corpus = data["tokenized_corpus"]
            logger.info("Loaded %d indexed chunks from disk.", len(self.chunks))
            return True
        except Exception as e:
            logger.error("Failed to load existing indexes: %s", e)
            return False

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense similarity search via FAISS."""
        if self.faiss_index is None or not self.chunks:
            return []
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        k = min(top_k, len(self.chunks))
        distances, indices = self.faiss_index.search(query_vec.astype(np.float32), k)
        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Sparse lexical search via BM25."""
        if self.bm25_index is None or not self.chunks:
            return []
        tokenized_query = self.tokenize(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:min(top_k, len(self.chunks))]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def hybrid_search(self, query: str, top_k: int = settings.FINAL_TOP_K) -> List[ScoredChunk]:
        """
        Executes dense and sparse search and merges results using
        Reciprocal Rank Fusion (RRF):
        RRF_Score = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
        """
        if not self.chunks:
            return []

        k_rrf = settings.RRF_K
        dense_results = self.dense_search(query, top_k=settings.TOP_K_DENSE)
        sparse_results = self.sparse_search(query, top_k=settings.TOP_K_SPARSE)

        rrf_scores: Dict[int, float] = {}
        dense_ranks: Dict[int, int] = {}
        sparse_ranks: Dict[int, int] = {}

        for rank, (idx, _) in enumerate(dense_results, start=1):
            dense_ranks[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank))

        for rank, (idx, _) in enumerate(sparse_results, start=1):
            sparse_ranks[idx] = rank
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank))

        sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        
        results: List[ScoredChunk] = []
        for idx in sorted_indices[:top_k]:
            results.append(ScoredChunk(
                chunk=self.chunks[idx],
                dense_rank=dense_ranks.get(idx),
                sparse_rank=sparse_ranks.get(idx),
                rrf_score=round(rrf_scores[idx], 6)
            ))

        return results
