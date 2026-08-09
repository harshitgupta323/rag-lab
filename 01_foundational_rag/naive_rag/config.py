from pathlib import Path


# -------------------------------------------------
# Project directories
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

VECTOR_DB_DIR = BASE_DIR / "chroma_db"

LOG_DIR = BASE_DIR / "logs"


# -------------------------------------------------
# Embeddings
# -------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------------------------
# Chunking
# -------------------------------------------------

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


# -------------------------------------------------
# Retrieval
# -------------------------------------------------

TOP_K = 4


# -------------------------------------------------
# LLM
# -------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"

TEMPERATURE = 0