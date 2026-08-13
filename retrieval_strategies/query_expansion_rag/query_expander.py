import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv
from langchain_groq import ChatGroq

from shared.utils.logging_config import setup_logging, get_logger

from config import GROQ_MODEL, QUERY_EXPANSION_TEMPERATURE, LOG_DIR


load_dotenv()

setup_logging(LOG_DIR)
logger = get_logger(__name__)


def get_llm():
    """Initialize the Groq LLM used for query expansion."""

    logger.info("Initializing Groq query expansion LLM | model=%s", GROQ_MODEL)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.warning("GROQ_API_KEY is not configured")
        raise ValueError("GROQ_API_KEY is not set")

    llm = ChatGroq(model=GROQ_MODEL, temperature=QUERY_EXPANSION_TEMPERATURE, api_key=api_key)

    logger.info("Groq query expansion LLM initialized successfully")

    return llm


def clean_expanded_query(response: str):
    """Clean the generated expanded query."""

    expanded_query = response.strip()

    expanded_query = expanded_query.replace("\n", " ")
    expanded_query = expanded_query.strip('"').strip("'").strip()

    if not expanded_query:
        raise ValueError("LLM returned an empty expanded query")

    return expanded_query


def expand_with_llm(query: str):
    """Expand the query using the Groq LLM."""

    logger.info("Generating expanded query using Groq")

    prompt = f"""You are a search query expansion system.

    Expand the following user query for semantic document retrieval.

    Add relevant:
    - Synonyms
    - Related terminology
    - Domain-specific terms
    - Important concepts implied by the query

    Do not answer the question.

    Do not rewrite it as another question.

    Return ONLY one expanded search query as a single line.

    Original query:
    {query}
    """

    llm = get_llm()
    response = llm.invoke(prompt)

    expanded_query = clean_expanded_query(response.content)

    logger.info("Successfully generated expanded query")

    return expanded_query


def expand_rule_based(query: str):
    """Generate a deterministic fallback expansion."""

    logger.info("Using rule-based query expansion fallback")

    expanded_query = f"{query} related information details policy guidelines requirements definitions"

    return expanded_query


def expand_query(query: str):
    """Expand the query using LLM with rule-based fallback."""

    logger.info("========== QUERY EXPANSION STARTED ==========")
    logger.debug("Original query: %s", query)

    try:
        expanded_query = expand_with_llm(query)

        logger.info("Query expansion completed using Groq")
        logger.debug("Expanded query: %s", expanded_query)
        logger.info("========== QUERY EXPANSION COMPLETED ==========")

        return expanded_query

    except Exception:
        logger.exception("LLM query expansion failed; using rule-based fallback")

        expanded_query = expand_rule_based(query)

        logger.info("Rule-based query expansion completed")
        logger.debug("Expanded query: %s", expanded_query)
        logger.info("========== QUERY EXPANSION COMPLETED ==========")

        return expanded_query