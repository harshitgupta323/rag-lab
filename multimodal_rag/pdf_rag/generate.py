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


def generate_answer(llm, query, context):
    if not context:
        return "I could not find relevant information in the PDF documents."

    prompt = f"""You are a PDF question-answering assistant.

    Answer the user's question using the retrieved PDF context.

    User Question:
    {query}

    Retrieved PDF Context:
    {context}

    Instructions:
    - Use the provided PDF context as the primary source.
    - Do not invent information.
    - If the answer is not available in the context, clearly say so.
    - When possible, mention the relevant page number.
    - Provide a concise and accurate answer.

    Answer:"""

    logger.info("Generating answer from PDF context")

    response = llm.invoke(prompt)

    logger.info("PDF RAG answer generated successfully")

    return response.content