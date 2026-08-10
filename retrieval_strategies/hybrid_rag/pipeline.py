import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.utils.logging_config import setup_logging, get_logger

from config import LOG_DIR

from ingest import ingest
from retrieve import retrieve_documents
from generate import generate_answer


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def initialize():
    """Initialize the Hybrid RAG system."""

    logger.info("========== HYBRID RAG INITIALIZATION STARTED ==========")

    try:
        ingest()
        logger.info("Hybrid RAG initialization completed successfully")

    except Exception:
        logger.exception("Hybrid RAG initialization failed")
        raise


def run_rag(query: str):
    """Execute the complete Hybrid RAG pipeline."""

    logger.info("========== HYBRID RAG QUERY STARTED ==========")
    logger.info("Query length: %d characters", len(query))

    try:
        documents = retrieve_documents(query)

        logger.info("Hybrid retrieval completed | documents=%d", len(documents))

        answer = generate_answer(query, documents)

        logger.info("Generation completed")
        logger.info("========== HYBRID RAG QUERY COMPLETED ==========")

        return {"answer": answer, "documents": documents}

    except Exception:
        logger.exception("Hybrid RAG pipeline failed")
        raise