from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "advanced_rag" / "agentic_rag" / "data"
CHROMA_DIR = PROJECT_ROOT / "advanced_rag" / "agentic_rag" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "advanced_rag" / "agentic_rag" / "logs"

COLLECTION_NAME = "agentic_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 5

MAX_AGENT_STEPS = 3

TEMPERATURE = 0