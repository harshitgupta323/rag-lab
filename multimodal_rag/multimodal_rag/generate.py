import os
from dotenv import load_dotenv
load_dotenv()

import logging

from langchain_groq import ChatGroq

from config import LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)


def create_llm():
    logger.info("Initializing multimodal LLM: %s", LLM_MODEL)

    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(llm, query, context):
    if not context:
        return "I could not find relevant information in the available multimodal documents."

    prompt = f"""You are a multimodal question-answering assistant.

    Answer the user's question using the retrieved information from multiple modalities.

    User Question:
    {query}

    Retrieved Context:
    {context}

    Instructions:
    - Use the retrieved context as your primary source.
    - The context can contain information originating from PDFs and images.
    - Do not invent facts.
    - If the information is insufficient, clearly state that.
    - Mention the relevant source when useful.
    - Provide a concise and accurate answer.

    Answer:"""

    logger.info("Generating multimodal answer")

    response = llm.invoke(prompt)

    logger.info("Multimodal answer generated successfully")

    return response.content