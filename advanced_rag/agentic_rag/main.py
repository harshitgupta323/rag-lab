import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from config import LOG_DIR
from shared.utils.logging_config import setup_logging
from advanced_rag.agentic_rag.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main():
    setup_logging(LOG_DIR)

    logger.info("Starting Agentic RAG")

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")

        if query.lower() == "exit":
            logger.info("Exiting Agentic RAG")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        try:
            answer = run_pipeline(query)

            print("\nAnswer:")
            print(answer)

        except Exception as e:
            logger.exception("Error while running Agentic RAG: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()