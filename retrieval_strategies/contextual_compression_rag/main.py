import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from shared.utils.logging_config import setup_logging
from pipeline import run_pipeline
from config import LOG_DIR
logger = logging.getLogger(__name__)


def main():
    setup_logging(LOG_DIR)
    logger.info("Starting Contextual Compression RAG")

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")

        if query.lower() == "exit":
            logger.info("Exiting Contextual Compression RAG")
            print("Exiting...")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        try:
            answer = run_pipeline(query)
            print("\nAnswer:")
            print(answer)
        except Exception:
            logger.exception("Error while running Contextual Compression RAG")
            print("An error occurred while processing your question. Check the logs for details.")


if __name__ == "__main__":
    main()