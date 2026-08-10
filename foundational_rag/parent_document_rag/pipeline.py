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
    """Initialize the Parent Document RAG system."""

    logger.info("========== PARENT DOCUMENT RAG INITIALIZATION STARTED ==========")

    try:
        retreiver = ingest()
        logger.info("Parent Document RAG initialization completed successfully")
        return retreiver

    except Exception:
        logger.exception("Parent Document RAG initialization failed")
        raise


def run_rag(retriever, query: str):
    """Execute the complete Parent Document RAG pipeline."""

    logger.info("========== PARENT DOCUMENT RAG QUERY STARTED ==========")
    logger.info("Query length: %d characters", len(query))

    try:
        documents = retrieve_documents(retriever, query)

        logger.info("Parent retrieval completed | documents=%d", len(documents))

        answer = generate_answer(query, documents)

        logger.info("Generation completed")
        logger.info("========== PARENT DOCUMENT RAG QUERY COMPLETED ==========")

        return {"answer": answer, "documents": documents}

    except Exception:
        logger.exception("Parent Document RAG pipeline failed")
        raise