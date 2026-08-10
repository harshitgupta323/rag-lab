from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200

CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 50

TOP_K = 4

GROQ_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0