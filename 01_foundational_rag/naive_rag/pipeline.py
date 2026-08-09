import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.utils.logging_config import (
    setup_logging,
    get_logger,
)

from config import LOG_DIR

from ingest import ingest
from retrieve import retrieve_documents
from generate import generate_answer


# -------------------------------------------------
# Logging
# -------------------------------------------------

setup_logging(LOG_DIR)

logger = get_logger(__name__)


def initialize():
    """
    Initialize the RAG system.
    """

    logger.info(
        "========== RAG INITIALIZATION STARTED =========="
    )

    try:

        ingest()

        logger.info(
            "RAG initialization completed successfully"
        )

    except Exception:

        logger.exception(
            "RAG initialization failed"
        )

        raise


def run_rag(query: str):
    """
    Execute the complete Naive RAG pipeline.
    """

    logger.info(
        "========== RAG QUERY STARTED =========="
    )

    logger.info(
        "Query length: %d characters",
        len(query),
    )

    try:

        # -----------------------------------------
        # Retrieval
        # -----------------------------------------

        documents = retrieve_documents(
            query
        )

        logger.info(
            "Retrieval completed | "
            "documents=%d",
            len(documents),
        )

        # -----------------------------------------
        # Generation
        # -----------------------------------------

        answer = generate_answer(
            query,
            documents,
        )

        logger.info(
            "Generation completed"
        )

        logger.info(
            "========== RAG QUERY COMPLETED =========="
        )

        return {
            "answer": answer,
            "documents": documents,
        }

    except Exception:

        logger.exception(
            "RAG pipeline failed"
        )

        raise