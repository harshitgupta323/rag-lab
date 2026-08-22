from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "multimodal_rag" / "multimodal_rag" / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
IMAGES_DIR = DATA_DIR / "images"

CHROMA_DIR = PROJECT_ROOT / "multimodal_rag" / "multimodal_rag" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "multimodal_rag" / "multimodal_rag" / "logs"

COLLECTION_NAME = "multimodal_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen/qwen3.6-27b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 4

TEMPERATURE = 0