import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    return vector_store


def retrieve_vector_context(vector_store, query):
    logger.info("Performing vector retrieval for query: %s", query)

    documents = vector_store.similarity_search_with_score(query, k=RETRIEVAL_TOP_K)

    logger.info("Retrieved %d vector documents", len(documents))

    if not documents:
        return ""

    return "\n\n".join(document.page_content for document, _ in documents)


def retrieve_graph_context(graph, entities):
    graph_context = []

    for entity in entities:
        relationships = graph.search_entity(entity)

        for source, relationship, target in relationships:
            graph_context.append(f"{source} --[{relationship}]--> {target}")

    logger.info("Retrieved %d graph relationships", len(graph_context))

    return "\n".join(graph_context)