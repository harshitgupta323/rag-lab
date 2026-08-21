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


def generate_direct_answer(llm, query):
    prompt = f"""You are a helpful question-answering assistant.

Answer the following question using your general knowledge.

Question:
{query}

Provide a concise and accurate answer."""

    logger.info("Generating direct answer without retrieval")

    response = llm.invoke(prompt)

    return response.content


def generate_rag_answer(llm, query, documents):
    if not documents:
        return "I could not find relevant information in the available documents."

    context = "\n\n".join(document.page_content for document, _ in documents)

    prompt = f"""You are a question-answering assistant.

    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {query}

    Instructions:
    - Use only information contained in the context.
    - Do not invent facts.
    - If the context does not contain enough information, say that you do not have enough information.
    - Provide a concise answer.

    Answer:"""

    logger.info("Generating RAG answer using %d documents", len(documents))

    response = llm.invoke(prompt)

    return response.content