import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from config import LOG_DIR
from shared.utils.logging_config import setup_logging
from graph import KnowledgeGraph

logger = logging.getLogger(__name__)


ENTITIES = [
    ("Harshit", "Person"),
    ("Senior Machine Learning Engineer", "Role"),
    ("Generative AI", "Technology"),
    ("Retrieval-Augmented Generation", "Technology"),
    ("Agentic AI", "Technology"),
    ("Machine Learning", "Technology"),
    ("Deep Learning", "Technology"),
    ("MLOps", "Technology"),
    ("LangChain", "Framework"),
    ("LangGraph", "Framework"),
    ("LangSmith", "Platform"),
    ("RAGAS", "Evaluation Tool"),
    ("MLflow", "Experiment Tracking Tool"),
    ("Chroma", "Vector Database"),
    ("Neo4j", "Graph Database"),
    ("Hugging Face", "Platform"),
    ("Groq", "LLM Platform"),
    ("Ollama", "LLM Platform"),
    ("Docker", "Technology"),
    ("RAG-Lab", "Project"),
    ("Naive RAG", "RAG"),
    ("Semantic RAG", "RAG"),
    ("Parent Document RAG", "RAG"),
    ("Hybrid RAG", "RAG"),
    ("Multi-Query RAG", "RAG"),
    ("Contextual RAG", "RAG"),
    ("Metadata Filtering RAG", "RAG"),
    ("Reranking RAG", "RAG"),
    ("Corrective RAG", "RAG"),
    ("Self-RAG", "RAG"),
    ("Adaptive RAG", "RAG"),
    ("Agentic RAG", "RAG"),
    ("Graph RAG", "RAG"),
]


RELATIONSHIPS = [
    ("Harshit", "WORKS_AS", "Senior Machine Learning Engineer"),
    ("Harshit", "WORKS_WITH", "Generative AI"),
    ("Harshit", "WORKS_WITH", "Retrieval-Augmented Generation"),
    ("Harshit", "WORKS_WITH", "Agentic AI"),
    ("Harshit", "WORKS_WITH", "Machine Learning"),
    ("Harshit", "WORKS_WITH", "Deep Learning"),
    ("Harshit", "WORKS_WITH", "MLOps"),
    ("Harshit", "WORKS_WITH", "LangChain"),
    ("Harshit", "WORKS_WITH", "LangGraph"),
    ("Harshit", "WORKS_WITH", "LangSmith"),
    ("Harshit", "WORKS_WITH", "RAGAS"),
    ("Harshit", "WORKS_WITH", "MLflow"),
    ("Harshit", "WORKS_WITH", "Chroma"),
    ("Harshit", "WORKS_WITH", "Neo4j"),
    ("Harshit", "WORKS_WITH", "Hugging Face"),
    ("Harshit", "WORKS_WITH", "Groq"),
    ("Harshit", "WORKS_WITH", "Ollama"),
    ("Harshit", "WORKS_WITH", "Docker"),

    ("RAG-Lab", "CONTAINS", "Naive RAG"),
    ("RAG-Lab", "CONTAINS", "Semantic RAG"),
    ("RAG-Lab", "CONTAINS", "Parent Document RAG"),
    ("RAG-Lab", "CONTAINS", "Hybrid RAG"),
    ("RAG-Lab", "CONTAINS", "Multi-Query RAG"),
    ("RAG-Lab", "CONTAINS", "Contextual RAG"),
    ("RAG-Lab", "CONTAINS", "Metadata Filtering RAG"),
    ("RAG-Lab", "CONTAINS", "Reranking RAG"),
    ("RAG-Lab", "CONTAINS", "Corrective RAG"),
    ("RAG-Lab", "CONTAINS", "Self-RAG"),
    ("RAG-Lab", "CONTAINS", "Adaptive RAG"),
    ("RAG-Lab", "CONTAINS", "Agentic RAG"),
    ("RAG-Lab", "CONTAINS", "Graph RAG"),

    ("Hybrid RAG", "COMBINES", "Dense Vector Retrieval"),
    ("Hybrid RAG", "COMBINES", "Lexical Retrieval"),

    ("Multi-Query RAG", "USES", "Multiple Query Variants"),

    ("Reranking RAG", "USES", "Document Reranking"),

    ("Corrective RAG", "PERFORMS", "Corrective Retrieval"),

    ("Self-RAG", "USES", "Self Reflection"),

    ("Adaptive RAG", "USES", "Dynamic Routing"),

    ("Agentic RAG", "USES", "Agent Decision Making"),

    ("Graph RAG", "REPRESENTS", "Entities"),
    ("Graph RAG", "REPRESENTS", "Relationships"),
    ("Graph RAG", "USES", "Neo4j"),

    ("RAG-Lab", "USES", "Chroma"),
    ("RAG-Lab", "USES", "RAGAS"),
    ("RAG-Lab", "USES", "MLflow"),
    ("RAG-Lab", "USES", "LangSmith"),
    ("RAG-Lab", "USES", "LangGraph"),
]


def populate_graph():
    logger.info("Starting Graph RAG knowledge graph population")

    graph = KnowledgeGraph()

    try:
        graph.verify_connection()

        graph.clear_graph()

        logger.info("Creating %d entities", len(ENTITIES))

        for name, entity_type in ENTITIES:
            graph.add_entity(name, entity_type)

        logger.info("Creating %d relationships", len(RELATIONSHIPS))

        for source, relationship, target in RELATIONSHIPS:
            graph.add_relationship(source, relationship, target)

        logger.info("Knowledge graph populated successfully")
        logger.info("Created %d entities and %d relationships", len(ENTITIES), len(RELATIONSHIPS))

    except Exception as e:
        logger.exception("Error while populating knowledge graph: %s", e)
        raise

    finally:
        graph.close()


if __name__ == "__main__":
    setup_logging(LOG_DIR)
    populate_graph()