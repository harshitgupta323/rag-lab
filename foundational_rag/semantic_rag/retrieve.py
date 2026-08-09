import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from shared.utils.logging_config import setup_logging, get_logger

from config import VECTOR_DB_DIR, LOG_DIR, EMBEDDING_MODEL, TOP_K


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def get_vector_store():
    """Load the existing Chroma vector store."""

    logger.info("Loading Chroma vector store from: %s", VECTOR_DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name="semantic_rag", persist_directory=str(VECTOR_DB_DIR), embedding_function=embeddings)

    logger.info("Semantic RAG vector store loaded successfully")

    return vector_store


def retrieve_documents(query: str, k: int = TOP_K):
    """Retrieve the top-k semantically similar chunks."""

    logger.info("Retrieving semantic chunks | top_k=%d", k)
    logger.debug("Query received: %s", query)

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(query, k=k)

    logger.info("Retrieved %d semantic chunk(s)", len(documents))

    return documents


if __name__ == "__main__":
    query = input("Enter your query: ")
    documents = retrieve_documents(query)

    print("\nRetrieved Documents:\n")

    for index, document in enumerate(documents, start=1):
        print(f"--- Document {index} ---")
        print(document.page_content)
        print("Metadata:", document.metadata)
        print()