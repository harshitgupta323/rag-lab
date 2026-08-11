import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.utils.logging_config import setup_logging, get_logger

from config import LOG_DIR

from ingest import ingest
from query_generator import generate_query_variants
from retrieve import retrieve_documents
from generate import generate_answer


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def initialize():
    """Initialize the Multi-Query RAG system."""

    logger.info("========== MULTI-QUERY RAG INITIALIZATION STARTED ==========")

    try:
        ingest()
        logger.info("Multi-Query RAG initialization completed successfully")

    except Exception:
        logger.exception("Multi-Query RAG initialization failed")
        raise


def run_rag(query: str):
    """Execute the complete Multi-Query RAG pipeline."""

    logger.info("========== MULTI-QUERY RAG QUERY STARTED ==========")
    logger.info("Query length: %d characters", len(query))

    try:
        query_variants = generate_query_variants(query)

        logger.info("Generated %d query variants", len(query_variants))

        documents = retrieve_documents(query_variants)

        logger.info("Multi-query retrieval completed | documents=%d", len(documents))

        answer = generate_answer(query, documents)

        logger.info("Generation completed")
        logger.info("========== MULTI-QUERY RAG QUERY COMPLETED ==========")

        return {"answer": answer, "documents": documents, "query_variants": query_variants}

    except Exception:
        logger.exception("Multi-Query RAG pipeline failed")
        raise