import logging

logger = logging.getLogger(__name__)


def classify_query(llm, query):
    prompt = f"""You are a query routing system for a Retrieval-Augmented Generation application.

    Classify the user query into exactly one of these categories:

    simple
    retrieval
    complex

    Definitions:

    simple:
    The question can be answered using general knowledge and does not require information from the local knowledge base.

    retrieval:
    The question asks about information that should be answered using the available knowledge base.

    complex:
    The question requires detailed, multi-part, or highly specific information and should use expanded retrieval.

    User Query:
    {query}

    Return only one label: simple, retrieval, or complex."""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()

    logger.info("Query classification result: %s", route)

    if route not in {"simple", "retrieval", "complex"}:
        logger.warning("Unexpected routing result: %s. Falling back to retrieval.", route)
        return "retrieval"

    return route