from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "advanced_rag" / "adaptive_rag" / "data"
CHROMA_DIR = PROJECT_ROOT / "advanced_rag" / "adaptive_rag" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "advanced_rag" / "adaptive_rag" / "logs"

COLLECTION_NAME = "adaptive_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 4
COMPLEX_RETRIEVAL_TOP_K = 8

TEMPERATURE = 0