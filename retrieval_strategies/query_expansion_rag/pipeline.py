import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.utils.logging_config import setup_logging, get_logger

from config import LOG_DIR

from ingest import ingest
from query_expander import expand_query
from retrieve import retrieve_documents
from generate import generate_answer


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def initialize():
    """Initialize the Query Expansion RAG system."""

    logger.info("========== QUERY EXPANSION RAG INITIALIZATION STARTED ==========")

    try:
        ingest()
        logger.info("Query Expansion RAG initialization completed successfully")

    except Exception:
        logger.exception("Query Expansion RAG initialization failed")
        raise


def run_rag(query: str):
    """Execute the complete Query Expansion RAG pipeline."""

    logger.info("========== QUERY EXPANSION RAG QUERY STARTED ==========")
    logger.info("Query length: %d characters", len(query))

    try:
        expanded_query = expand_query(query)

        logger.info("Query expansion completed")

        documents = retrieve_documents(query, expanded_query)

        logger.info("Query expansion retrieval completed | documents=%d", len(documents))

        answer = generate_answer(query, documents)

        logger.info("Generation completed")
        logger.info("========== QUERY EXPANSION RAG QUERY COMPLETED ==========")

        return {"answer": answer, "documents": documents, "expanded_query": expanded_query}

    except Exception:
        logger.exception("Query Expansion RAG pipeline failed")
        raise