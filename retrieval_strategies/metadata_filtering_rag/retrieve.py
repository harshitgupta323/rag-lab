import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    return vector_store


def build_metadata_filter(metadata_filter):
    if not metadata_filter:
        return None

    conditions = []

    for key, value in metadata_filter.items():
        conditions.append({key: {"$eq": value}})

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def retrieve_documents(vector_store, query, metadata_filter=None):
    logger.info("Retrieving documents for query: %s", query)

    chroma_filter = build_metadata_filter(metadata_filter)

    if chroma_filter:
        logger.info("Applying Chroma metadata filter: %s", chroma_filter)
        documents = vector_store.similarity_search(query, k=RETRIEVAL_TOP_K, filter=chroma_filter)
    else:
        logger.info("No metadata filter applied")
        documents = vector_store.similarity_search(query, k=RETRIEVAL_TOP_K)

    logger.info("Retrieved %d documents", len(documents))

    return documents