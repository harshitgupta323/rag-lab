import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    return vector_store


def retrieve_documents(vector_store, query):
    logger.info("Retrieving documents for query: %s", query)

    documents = vector_store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)

    logger.info("Retrieved %d PDF chunks", len(documents))

    return documents


def build_context(documents):
    if not documents:
        return ""

    context_parts = []

    for document, score in documents:
        source = document.metadata.get("source", "Unknown")
        page = document.metadata.get("page", "Unknown")

        context_parts.append(f"Source: {source}, Page: {page}\n{document.page_content}")

    return "\n\n".join(context_parts)