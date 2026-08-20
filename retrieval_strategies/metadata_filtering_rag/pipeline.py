import logging

from ingest import ingest_documents
from retrieve import create_vector_store, retrieve_documents
from generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query, metadata_filter=None):
    logger.info("Starting Metadata Filtering RAG pipeline")

    ingest_documents()

    vector_store = create_vector_store()
    llm = create_llm()

    documents = retrieve_documents(vector_store, query, metadata_filter)

    answer = generate_answer(llm, query, documents)

    logger.info("Metadata Filtering RAG pipeline completed successfully")

    return answer