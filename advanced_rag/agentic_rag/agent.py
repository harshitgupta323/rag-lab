import logging

from tools import retrieve_information

from config import MAX_AGENT_STEPS

logger = logging.getLogger(__name__)


def decide_action(llm, query, context):
    prompt = f"""You are the decision-making agent in an Agentic RAG system.

    Your job is to decide what action should be taken next.

    User Query:
    {query}

    Current Retrieved Context:
    {context if context else "No context has been retrieved yet."}

    Choose exactly one action:

    retrieve
    refine
    answer

    Definitions:

    retrieve:
    Use the knowledge base to retrieve information relevant to the query.

    refine:
    The current retrieved information is insufficient or poorly targeted. A refined search query should be generated.

    answer:
    Enough information is available to answer the user's question.

    Return only one action."""

    response = llm.invoke(prompt)
    action = response.content.strip().lower()

    logger.info("Agent selected action: %s", action)

    if action not in {"retrieve", "refine", "answer"}:
        logger.warning("Unexpected agent action: %s. Falling back to retrieve.", action)
        return "retrieve"

    return action


def refine_query(llm, query, context):
    prompt = f"""You are a query refinement agent.

    The original user query is:

    {query}

    The currently retrieved context is:

    {context if context else "No useful context was retrieved."}

    Create a better search query that can retrieve information needed to answer the original question.

    Return only the refined search query."""

    response = llm.invoke(prompt)
    refined_query = response.content.strip()

    logger.info("Agent generated refined query: %s", refined_query)

    return refined_query


def run_agent(llm, vector_store, query):
    context = ""
    current_query = query

    for step in range(1, MAX_AGENT_STEPS + 1):
        logger.info("Agent step %d/%d", step, MAX_AGENT_STEPS)

        action = decide_action(llm, query, context)

        if action == "retrieve":
            retrieved_context = retrieve_information(vector_store, current_query)

            if context:
                context = context + "\n\n" + retrieved_context
            else:
                context = retrieved_context

        elif action == "refine":
            current_query = refine_query(llm, query, context)
            retrieved_context = retrieve_information(vector_store, current_query)

            if context:
                context = context + "\n\n" + retrieved_context
            else:
                context = retrieved_context

        elif action == "answer":
            logger.info("Agent decided that sufficient information is available")
            break

    logger.info("Agent execution completed after %d step(s)", step)

    return context