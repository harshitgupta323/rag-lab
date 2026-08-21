import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    return vector_store


def retrieve_information(vector_store, query):
    logger.info("Agent retrieval tool called with query: %s", query)

    documents = vector_store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)

    logger.info("Agent retrieval tool returned %d documents", len(documents))

    if not documents:
        return "No relevant documents were found."

    context = "\n\n".join(document.page_content for document, _ in documents)

    return context