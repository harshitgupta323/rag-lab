import logging
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

from config import LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)


def create_llm():
    logger.info("Initializing LLM: %s", LLM_MODEL)

    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(llm, query, documents):
    if not documents:
        logger.warning("No relevant documents found")

        return "I could not find relevant information matching the specified filters."

    context = "\n\n".join(document.page_content for document in documents)

    prompt = f"""You are a helpful question-answering assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say that you do not have enough information.

Answer:"""

    logger.info("Generating answer using %d documents", len(documents))

    response = llm.invoke(prompt)

    logger.info("Answer generated successfully")

    return response.content