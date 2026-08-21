from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "advanced_rag" / "graph_rag" / "data"
CHROMA_DIR = PROJECT_ROOT / "advanced_rag" / "graph_rag" / "chroma_db"
LOG_DIR = PROJECT_ROOT / "advanced_rag" / "graph_rag" / "logs"

COLLECTION_NAME = "graph_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_TOP_K = 4

NEO4J_URI = "bolt://localhost:7687"
NEO4J_DATABASE = "neo4j"
NEO4J_PASSWORD = "Password"

TEMPERATURE = 0