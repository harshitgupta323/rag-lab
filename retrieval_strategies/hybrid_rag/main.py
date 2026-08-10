import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.utils.logging_config import setup_logging, get_logger

from config import LOG_DIR

from pipeline import initialize, run_rag


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def main():

    logger.info("Starting Hybrid RAG application")

    print("=" * 60)
    print("                RAG LAB - HYBRID RAG")
    print("=" * 60)

    try:
        initialize()

    except Exception:
        logger.exception("Application initialization failed")
        print("\nFailed to initialize Hybrid RAG system.")
        return

    while True:

        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            logger.info("User exited the application")
            print("Goodbye!")
            break

        if not query.strip():
            logger.warning("Empty query received")
            continue

        try:
            logger.info("Processing user query")

            result = run_rag(query)

            print("\n" + "=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result["answer"])

            print("\n" + "=" * 60)
            print("RETRIEVED HYBRID CONTEXT")
            print("=" * 60)

            for index, document in enumerate(result["documents"], start=1):
                print(f"\n--- Hybrid Chunk {index} ---")
                print(document.page_content)
                print("Source:", document.metadata.get("source"))

        except Exception:
            logger.exception("Failed to process user query")
            print("\nAn error occurred while processing your query.")


if __name__ == "__main__":
    main()