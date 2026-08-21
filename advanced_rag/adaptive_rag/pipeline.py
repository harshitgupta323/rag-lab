import logging

from ingest import ingest_documents
from retrieve import create_vector_store, retrieve_documents
from route import classify_query
from generate import create_llm, generate_direct_answer, generate_rag_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Adaptive RAG pipeline")

    ingest_documents()

    vector_store = create_vector_store()
    llm = create_llm()

    route = classify_query(llm, query)

    logger.info("Selected route: %s", route)

    if route == "simple":
        answer = generate_direct_answer(llm, query)

    elif route == "retrieval":
        documents = retrieve_documents(vector_store, query)
        answer = generate_rag_answer(llm, query, documents)

    else:
        documents = retrieve_documents(vector_store, query, complex_query=True)
        answer = generate_rag_answer(llm, query, documents)

    logger.info("Adaptive RAG pipeline completed successfully")

    return answer