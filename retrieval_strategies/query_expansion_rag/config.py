from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
VECTOR_DB_DIR = BASE_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 5
FINAL_TOP_K = 4

RRF_K = 60

GROQ_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0
QUERY_EXPANSION_TEMPERATURE = 0