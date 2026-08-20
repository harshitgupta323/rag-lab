import logging
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from config import LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)


def create_compressor():
    logger.info("Initializing compression LLM: %s", LLM_MODEL)
    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY"))


def compress_documents(llm, query, documents):
    compressed_documents = []

    logger.info("Compressing %d retrieved documents", len(documents))

    for document in documents:
        logger.info("Original document - %s", document)
        prompt = f"""You are a contextual compression system.

        Your task is to extract relevant text from the DOCUMENT that can help answer the QUESTION.

        QUESTION:
        {query}

        DOCUMENT:
        {document.page_content}

        RULES:
        1. Return only relevant sentences copied from the DOCUMENT.
        2. Do not summarize.
        3. Do not rewrite the sentences.
        4. Do not return document numbers.
        5. Do not return indexes.
        6. Do not return ranges such as 0..4 or 1-5.
        7. Do not return JSON.
        8. Do not return metadata.
        9. Do not explain your answer.
        10. If nothing in the DOCUMENT is relevant, return exactly: NO_RELEVANT_CONTENT

        RELEVANT TEXT:"""

        response = llm.invoke(prompt)
        logger.info("compression response: %s", response)
        content = response.content.strip()

        logger.info("Compression response: %s", content)

        if not content:
            continue

        if content.upper() == "NO_RELEVANT_CONTENT":
            continue

        if content in {"0..4", "0-4", "0,1,2,3,4"}:
            logger.warning("Invalid compression output detected: %s", content)
            continue

        compressed_documents.append(Document(page_content=content, metadata=document.metadata))

    logger.info("Compression produced %d documents", len(compressed_documents))

    return compressed_documents