import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from shared.utils.logging_config import setup_logging, get_logger
from retrieval_strategies.reranking_rag.pipeline import build_pipeline, run_pipeline

from config import LOG_DIR

logger = get_logger(__name__)


def main():
    setup_logging(LOG_DIR)
    logger.info("Starting Reranking RAG application")

    retriever, reranker, llm = build_pipeline()

    query = input("Enter your question: ")

    answer, retrieved_documents, reranked_documents = run_pipeline(query, retriever, reranker, llm)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print("\n" + "=" * 80)
    print("INITIAL RETRIEVAL")
    print("=" * 80)

    for index, document in enumerate(retrieved_documents, start=1):
        source = document.metadata.get("source", "Unknown")
        print(f"{index}. {source}")

    print("\n" + "=" * 80)
    print("RERANKED DOCUMENTS")
    print("=" * 80)

    for index, document in enumerate(reranked_documents, start=1):
        score = document.metadata.get("rerank_score", 0.0)
        source = document.metadata.get("source", "Unknown")
        print(f"{index}. Score: {score:.4f} | Source: {source}")


if __name__ == "__main__":
    main()