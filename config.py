import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseModel):
    PROJECT_NAME: str = "Enterprise Secure & Compliant RAG Platform"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Storage Paths
    DATA_DIR: Path = BASE_DIR / "data"
    STORAGE_DIR: Path = BASE_DIR / "storage"
    FAISS_INDEX_DIR: Path = STORAGE_DIR / "faiss_index"
    BM25_INDEX_PATH: Path = STORAGE_DIR / "bm25_index.pkl"
    AUDIT_LOG_PATH: Path = STORAGE_DIR / "audit_log.jsonl"

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384

    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Hybrid Retrieval Configuration
    TOP_K_DENSE: int = int(os.getenv("TOP_K_DENSE", "4"))
    TOP_K_SPARSE: int = int(os.getenv("TOP_K_SPARSE", "4"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    FINAL_TOP_K: int = int(os.getenv("FINAL_TOP_K", "3"))

    # Ollama Local LLM Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_REQUEST_TIMEOUT: float = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "60.0"))

    # PII Configuration
    CONFIDENCE_THRESHOLD: float = float(os.getenv("PII_CONFIDENCE_THRESHOLD", "0.6"))

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
