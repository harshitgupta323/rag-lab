import logging

from ingest import ingest_documents
from retrieve import create_vector_store, retrieve_documents
from evaluate_retrieval import evaluate_retrieval
from generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Corrective RAG pipeline")

    ingest_documents()

    vector_store = create_vector_store()
    llm = create_llm()

    retrieved_documents = retrieve_documents(vector_store, query)

    evaluation = evaluate_retrieval(llm, query, retrieved_documents)

    logger.info("Retrieval evaluation: %s", evaluation)

    if evaluation == "correct":
        logger.info("Retrieved context considered correct")

        answer = generate_answer(llm, query, retrieved_documents)

    elif evaluation == "ambiguous":
        logger.info("Retrieved context considered ambiguous, generating answer with available context")

        answer = generate_answer(llm, query, retrieved_documents)

    else:
        logger.warning("Retrieved context considered incorrect")

        answer = "The initial retrieval did not provide sufficiently relevant information to answer the question."

    logger.info("Corrective RAG pipeline completed successfully")

    return answer