import logging

logger = logging.getLogger(__name__)


def evaluate_retrieval(llm, query, documents):
    if not documents:
        logger.warning("No documents available for retrieval evaluation")
        return "incorrect"

    context = "\n\n".join([document.page_content for document, _ in documents])

    prompt = f"""You are a retrieval evaluation system.

    Determine whether the retrieved context contains enough relevant information to answer the question.

    Question:
    {query}

    Retrieved Context:
    {context}

    Return exactly one of these labels:
    correct
    incorrect
    ambiguous

    Definitions:
    correct = The retrieved context contains sufficient relevant information to answer the question.
    incorrect = The retrieved context is irrelevant or does not contain useful information.
    ambiguous = The retrieved context contains some potentially useful information but may be incomplete.

    Return only the label."""

    response = llm.invoke(prompt)
    result = response.content.strip().lower()

    logger.info("Retrieval evaluation result: %s", result)

    if result not in {"correct", "incorrect", "ambiguous"}:
        logger.warning("Unexpected retrieval evaluation result: %s", result)
        return "ambiguous"

    return result