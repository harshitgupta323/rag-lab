import logging

from ingest import ingest_documents
from retrieve import create_vector_store, retrieve_documents
from evaluate import evaluate_context, evaluate_answer
from generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Self-RAG pipeline")

    ingest_documents()

    vector_store = create_vector_store()
    llm = create_llm()

    retrieved_documents = retrieve_documents(vector_store, query)

    context_evaluation = evaluate_context(llm, query, retrieved_documents)

    if context_evaluation == "not_relevant":
        logger.warning("Retrieved context was evaluated as not relevant")
        return "I could not find sufficiently relevant information in the available documents."

    logger.info("Retrieved context was evaluated as relevant")

    answer = generate_answer(llm, query, retrieved_documents)

    context = "\n\n".join(document.page_content for document, _ in retrieved_documents)

    answer_evaluation = evaluate_answer(llm, query, context, answer)

    if answer_evaluation == "unsupported":
        logger.warning("Generated answer was evaluated as unsupported")
        return "The retrieved information was not sufficient to produce a reliable answer."

    logger.info("Generated answer was evaluated as supported")
    logger.info("Self-RAG pipeline completed successfully")

    return answer