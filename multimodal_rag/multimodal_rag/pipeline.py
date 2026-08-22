import logging

from ingest import ingest_documents
from retrieve import create_vector_store, retrieve_documents, build_context
from generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Multimodal RAG pipeline")

    ingest_documents()

    vector_store = create_vector_store()
    llm = create_llm()

    documents = retrieve_documents(vector_store, query)

    context = build_context(documents)

    answer = generate_answer(llm, query, context)

    logger.info("Multimodal RAG pipeline completed successfully")

    return answer