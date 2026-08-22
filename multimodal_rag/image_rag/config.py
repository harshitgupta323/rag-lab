from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "multimodal_rag" / "image_rag" / "data"
CHROMA_DIR = PROJECT_ROOT / "multimodal_rag" / "image_rag" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "multimodal_rag" / "image_rag" / "logs"

COLLECTION_NAME = "image_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3.6-27b"

RETRIEVAL_TOP_K = 3

TEMPERATURE = 0