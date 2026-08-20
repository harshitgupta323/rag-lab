from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "retrieval_strategies" / "metadata_filtering_rag" / "data"
CHROMA_DIR = PROJECT_ROOT / "retrieval_strategies" / "metadata_filtering_rag" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "retrieval_strategies" / "metadata_filtering_rag" / "logs"

COLLECTION_NAME = "metadata_filtering_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3.6-27b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 5

TEMPERATURE = 0