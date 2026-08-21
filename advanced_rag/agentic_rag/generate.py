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
        return "I could not find relevant information in the available documents."

    prompt = f"""You are a question-answering assistant operating inside an Agentic RAG system.

    Answer the user's question using the retrieved context.

    User Question:
    {query}

    Retrieved Context:
    {context}

    Instructions:
    - Use the retrieved context as the primary source of information.
    - Do not invent information.
    - If the context does not contain enough information, clearly say so.
    - Provide a concise and accurate answer.

    Answer:"""

    logger.info("Generating final answer from agent-selected context")

    response = llm.invoke(prompt)

    logger.info("Final answer generated successfully")

    return response.content