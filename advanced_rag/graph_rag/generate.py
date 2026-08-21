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


def extract_entities(llm, query):
    prompt = f"""Extract the important named entities from the following question.

    Question:
    {query}

    Return only the entities separated by commas.

    If there are no identifiable entities, return NONE."""

    response = llm.invoke(prompt)

    result = response.content.strip()

    if result.upper() == "NONE":
        return []

    entities = [entity.strip() for entity in result.split(",") if entity.strip()]

    logger.info("Extracted entities: %s", entities)

    return entities


def generate_answer(llm, query, vector_context, graph_context):
    if not vector_context and not graph_context:
        return "I could not find relevant information in the available knowledge sources."

    prompt = f"""You are a question-answering assistant using Graph RAG.

    Answer the user's question using the provided vector context and knowledge graph context.

    Question:
    {query}

    Vector Context:
    {vector_context if vector_context else "No vector context available."}

    Knowledge Graph Context:
    {graph_context if graph_context else "No graph context available."}

    Instructions:
    - Use the provided information as your primary source.
    - Use relationships from the knowledge graph when they help answer the question.
    - Do not invent facts.
    - If the available information is insufficient, clearly state that.
    - Provide a concise and accurate answer.

    Answer:"""

    logger.info("Generating final Graph RAG answer")

    response = llm.invoke(prompt)

    logger.info("Graph RAG answer generated successfully")

    return response.content