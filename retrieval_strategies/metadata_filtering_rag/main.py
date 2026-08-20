import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

from config import LOG_DIR

from shared.utils.logging_config import setup_logging
from pipeline import run_pipeline

logger = logging.getLogger(__name__)


def main():
    setup_logging(LOG_DIR)

    logger.info("Starting Metadata Filtering RAG")

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")

        if query.lower() == "exit":
            logger.info("Exiting Metadata Filtering RAG")
            break

        if not query.strip():
            print("Please enter a valid question.")
            continue

        category = input("Enter category filter (or press Enter to skip): ").strip()
        year = input("Enter year filter (or press Enter to skip): ").strip()

        metadata_filter = {}

        if category:
            metadata_filter["category"] = category

        if year:
            try:
                metadata_filter["year"] = int(year)
            except ValueError:
                print("Year must be a number.")
                continue

        if not metadata_filter:
            metadata_filter = None

        try:
            answer = run_pipeline(query, metadata_filter)

            print("\nAnswer:")
            print(answer)

        except Exception as e:
            logger.exception("Error while running Metadata Filtering RAG: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()