import logging

from ingest import ingest_documents
from graph import KnowledgeGraph
from retrieve import create_vector_store, retrieve_vector_context, retrieve_graph_context
from generate import create_llm, extract_entities, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Graph RAG pipeline")

    ingest_documents()

    vector_store = create_vector_store()
    llm = create_llm()
    graph = KnowledgeGraph()

    try:
        graph.verify_connection()

        entities = extract_entities(llm, query)

        vector_context = retrieve_vector_context(vector_store, query)
        graph_context = retrieve_graph_context(graph, entities)

        answer = generate_answer(llm, query, vector_context, graph_context)

        logger.info("Graph RAG pipeline completed successfully")

        return answer

    finally:
        graph.close()