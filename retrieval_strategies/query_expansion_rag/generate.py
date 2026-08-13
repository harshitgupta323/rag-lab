import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dotenv import load_dotenv
from langchain_groq import ChatGroq

from shared.utils.logging_config import setup_logging, get_logger

from config import GROQ_MODEL, TEMPERATURE, LOG_DIR


load_dotenv()

setup_logging(LOG_DIR)
logger = get_logger(__name__)


def get_llm():
    """Initialize the Groq answer generation LLM."""

    logger.info("Initializing Groq answer generation LLM | model=%s", GROQ_MODEL)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        logger.error("GROQ_API_KEY is not configured")
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    llm = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE, api_key=api_key)

    logger.info("Groq answer generation LLM initialized successfully")

    return llm


def generate_answer(query: str, documents):
    """Generate an answer using the retrieved context."""

    logger.info("Starting answer generation | context_documents=%d", len(documents))

    context = "\n\n".join(document.page_content for document in documents)

    prompt = f"""You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Do not invent information.

Context:
----------------
{context}
----------------

Question:
{query}

Answer:
"""

    try:
        llm = get_llm()

        logger.info("Sending answer generation request to Groq | model=%s", GROQ_MODEL)

        response = llm.invoke(prompt)

        logger.info("LLM answer generation completed successfully")

        return response.content

    except Exception:
        logger.exception("LLM answer generation failed")
        raise