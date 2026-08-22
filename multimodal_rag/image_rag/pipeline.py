import logging

from ingest import ingest_images
from retrieve import create_vector_store, retrieve_images, build_context
from generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Image RAG pipeline")

    ingest_images()

    vector_store = create_vector_store()
    llm = create_llm()

    documents = retrieve_images(vector_store, query)

    context = build_context(documents)

    answer = generate_answer(llm, query, context)

    logger.info("Image RAG pipeline completed successfully")

    return answer