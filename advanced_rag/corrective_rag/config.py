from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "advanced_rag" / "corrective_rag" / "data"
CHROMA_DIR = PROJECT_ROOT / "advanced_rag" / "corrective_rag" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "advanced_rag" / "corrective_rag" / "logs"

COLLECTION_NAME = "corrective_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3.6-27b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 5

TEMPERATURE = 0