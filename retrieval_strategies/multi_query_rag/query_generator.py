import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv
from langchain_groq import ChatGroq

from shared.utils.logging_config import setup_logging, get_logger

from config import GROQ_MODEL, QUERY_GENERATION_TEMPERATURE, NUMBER_OF_QUERY_VARIANTS, LOG_DIR


load_dotenv()

setup_logging(LOG_DIR)
logger = get_logger(__name__)


def get_llm():
    """Initialize the Groq LLM used for query generation."""

    logger.info("Initializing Groq query generation LLM | model=%s", GROQ_MODEL)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.warning("GROQ_API_KEY is not configured")
        raise ValueError("GROQ_API_KEY is not set")

    llm = ChatGroq(model=GROQ_MODEL, temperature=QUERY_GENERATION_TEMPERATURE, api_key=api_key)

    logger.info("Groq query generation LLM initialized successfully")

    return llm


def clean_queries(response: str, original_query: str):
    """Clean and validate generated query variants."""

    lines = response.splitlines()
    queries = []

    for line in lines:
        query = line.strip()
        query = query.lstrip("0123456789.-) ")
        query = query.strip('"').strip("'").strip()

        if query and query.lower() != original_query.lower() and query not in queries:
            queries.append(query)

    return queries[:NUMBER_OF_QUERY_VARIANTS]


def generate_with_llm(query: str):
    """Generate query variants using the Groq LLM."""

    logger.info("Generating %d query variants using Groq", NUMBER_OF_QUERY_VARIANTS)

    prompt = f"""You are a search query generation system.

Generate exactly {NUMBER_OF_QUERY_VARIANTS} different search queries for the user's question.

Each query should approach the information need from a different perspective.

Use:
- Different wording
- Related terminology
- Different aspects of the question
- More specific or broader formulations where useful

Return ONLY the queries, one per line.

Do not number them.

Original question:
{query}
"""

    llm = get_llm()
    response = llm.invoke(prompt)

    queries = clean_queries(response.content, query)

    if len(queries) < NUMBER_OF_QUERY_VARIANTS:
        logger.warning("LLM generated only %d valid query variants", len(queries))
        raise ValueError("LLM did not generate enough query variants")

    logger.info("Successfully generated %d query variants", len(queries))

    return queries


def generate_rule_based(query: str):
    """Generate deterministic fallback query variants."""

    logger.info("Using rule-based query generation fallback")

    variants = [
        query,
        f"What information is available about {query}",
        f"Explain {query}",
        f"Details regarding {query}",
        f"What does the documentation say about {query}",
    ]

    unique_variants = []

    for variant in variants:
        if variant not in unique_variants:
            unique_variants.append(variant)

    return unique_variants[:NUMBER_OF_QUERY_VARIANTS]


def generate_query_variants(query: str):
    """Generate query variants using LLM with rule-based fallback."""

    logger.info("========== QUERY GENERATION STARTED ==========")
    logger.debug("Original query: %s", query)

    try:
        queries = generate_with_llm(query)
        logger.info("Query generation completed using Groq")
        logger.info("========== QUERY GENERATION COMPLETED ==========")
        return queries

    except Exception:
        logger.exception("LLM query generation failed; using rule-based fallback")

        queries = generate_rule_based(query)

        logger.info("Generated %d fallback query variants", len(queries))
        logger.info("========== QUERY GENERATION COMPLETED ==========")

        return queries