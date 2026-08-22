import logging

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    return vector_store


def retrieve_documents(vector_store, query):
    logger.info("Retrieving multimodal documents for query: %s", query)

    documents = vector_store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)

    logger.info("Retrieved %d multimodal documents", len(documents))

    return documents


def build_context(documents):
    if not documents:
        return ""

    context_parts = []

    for document, score in documents:
        source = document.metadata.get("source", "Unknown")
        modality = document.metadata.get("modality", "Unknown")
        content = document.page_content

        context_parts.append(f"Source: {source}\nModality: {modality}\nContent: {content}")

    return "\n\n".join(context_parts)