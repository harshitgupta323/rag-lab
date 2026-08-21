import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_TOP_K, COMPLEX_RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    return vector_store


def retrieve_documents(vector_store, query, complex_query=False):
    top_k = COMPLEX_RETRIEVAL_TOP_K if complex_query else RETRIEVAL_TOP_K

    logger.info("Retrieving %d documents for query: %s", top_k, query)

    documents = vector_store.similarity_search_with_score(query, k=top_k)

    logger.info("Retrieved %d documents", len(documents))

    return documents