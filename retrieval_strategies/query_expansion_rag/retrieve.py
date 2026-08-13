import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from shared.utils.logging_config import setup_logging, get_logger

from config import VECTOR_DB_DIR, LOG_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K, FINAL_TOP_K, RRF_K


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def get_vector_store():
    """Load the Chroma vector store."""

    logger.info("Loading Chroma vector store from: %s", VECTOR_DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name="query_expansion_rag",
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )

    logger.info("Chroma vector store loaded successfully")

    return vector_store


def retrieve_for_query(query: str):
    """Retrieve documents for a single query."""

    logger.info("Running dense retrieval | top_k=%d", RETRIEVAL_TOP_K)
    logger.debug("Retrieval query: %s", query)

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(query, k=RETRIEVAL_TOP_K)

    logger.info("Retrieved %d document(s)", len(documents))

    return documents


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Combine ranked result lists using Reciprocal Rank Fusion."""

    logger.info("Applying Reciprocal Rank Fusion | rrf_k=%d", k)

    scores = {}
    documents = {}

    for result_list in result_lists:
        for rank, document in enumerate(result_list, start=1):
            document_id = document.metadata.get("chunk_id")

            if not document_id:
                document_id = f"{document.metadata.get('source', '')}:{document.page_content}"

            if document_id not in scores:
                scores[document_id] = 0.0
                documents[document_id] = document

            scores[document_id] += 1 / (k + rank)

    ranked_documents = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    fused_documents = [documents[document_id] for document_id, _ in ranked_documents]

    logger.info("RRF produced %d unique document(s)", len(fused_documents))

    return fused_documents


def retrieve_documents(original_query: str, expanded_query: str):
    """Retrieve documents using the original and expanded queries."""

    logger.info("========== QUERY EXPANSION RETRIEVAL STARTED ==========")

    original_documents = retrieve_for_query(original_query)
    expanded_documents = retrieve_for_query(expanded_query)

    fused_documents = reciprocal_rank_fusion([original_documents, expanded_documents])

    final_documents = fused_documents[:FINAL_TOP_K]

    logger.info("Final retrieval returned %d document(s)", len(final_documents))
    logger.info("========== QUERY EXPANSION RETRIEVAL COMPLETED ==========")

    return final_documents