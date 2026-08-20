from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "qwen/qwen3.6-27b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 10
RERANKED_TOP_K = 3

TEMPERATURE = 0