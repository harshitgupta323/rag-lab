import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from langchain_groq import ChatGroq
from retrieval_strategies.reranking_rag.config import LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)

import os
from dotenv import load_dotenv

load_dotenv()


def create_llm():
    logger.info("Initializing LLM: %s", LLM_MODEL)
    groq_api_key = os.getenv("GROQ_API_KEY")
    return ChatGroq(api_key=groq_api_key, model=LLM_MODEL, temperature=TEMPERATURE)


def generate_answer(llm, query, documents):
    context = "\n\n".join(document.page_content for document in documents)

    prompt = f"""You are a helpful question-answering assistant.

        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {query}

        Instructions:
        - Use only information available in the context.
        - Do not make up information.
        - If the answer cannot be determined from the context, say that you do not have enough information.
        - Keep the answer clear and concise.

        Answer:"""

    logger.info("Generating answer using %d reranked documents", len(documents))

    response = llm.invoke(prompt)

    logger.info("Answer generated successfully")

    return response.content