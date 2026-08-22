import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    return vector_store


def retrieve_images(vector_store, query):
    logger.info("Retrieving images for query: %s", query)

    documents = vector_store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)

    logger.info("Retrieved %d image descriptions", len(documents))

    return documents


def build_context(documents):
    if not documents:
        return ""

    context_parts = []

    for document, score in documents:
        source = document.metadata.get("source", "Unknown")
        description = document.page_content

        context_parts.append(f"Image: {source}\nDescription: {description}\nSimilarity Score: {score}")

    return "\n\n".join(context_parts)