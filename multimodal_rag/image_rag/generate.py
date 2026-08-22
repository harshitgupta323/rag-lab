import logging
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

from config import LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)


def create_llm():
    logger.info("Initializing vision LLM: %s", LLM_MODEL)

    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(llm, query, context):
    if not context:
        return "I could not find relevant images for the query."

    prompt = f"""You are an image question-answering assistant.

    Answer the user's question using the retrieved image information.

    User Question:
    {query}

    Retrieved Image Information:
    {context}

    Instructions:
    - Use the retrieved image descriptions as your primary source.
    - Do not invent visual information.
    - If the retrieved information is insufficient, clearly state that.
    - Mention the relevant image filename when useful.
    - Provide a concise and accurate answer.

    Answer:"""

    logger.info("Generating Image RAG answer")

    response = llm.invoke(prompt)

    logger.info("Image RAG answer generated successfully")

    return response.content