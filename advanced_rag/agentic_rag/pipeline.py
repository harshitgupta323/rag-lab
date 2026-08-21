import logging

from ingest import ingest_documents
from tools import create_vector_store
from agent import run_agent
from generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Agentic RAG pipeline")

    ingest_documents()

    vector_store = create_vector_store()
    llm = create_llm()

    context = run_agent(llm, vector_store, query)

    answer = generate_answer(llm, query, context)

    logger.info("Agentic RAG pipeline completed successfully")

    return answer