import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from shared.utils.logging_config import setup_logging, get_logger

from config import VECTOR_DB_DIR, BM25_INDEX_PATH, LOG_DIR, EMBEDDING_MODEL, DENSE_TOP_K, BM25_TOP_K, FINAL_TOP_K, RRF_K


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def get_dense_retriever():
    """Load the dense Chroma retriever."""

    logger.info("Loading dense vector store")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name="hybrid_rag",
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": DENSE_TOP_K})

    logger.info("Dense retriever loaded successfully")

    return retriever


def get_bm25_index():
    """Load the persisted BM25 index."""

    logger.info("Loading BM25 index from: %s", BM25_INDEX_PATH)

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}")

    with open(BM25_INDEX_PATH, "rb") as file:
        index_data = pickle.load(file)

    logger.info("BM25 index loaded successfully")

    return index_data["bm25"], index_data["documents"]


def dense_retrieve(query: str):
    """Retrieve documents using dense semantic search."""

    logger.info("Running dense retrieval | top_k=%d", DENSE_TOP_K)

    retriever = get_dense_retriever()

    documents = retriever.invoke(query)

    logger.info("Dense retrieval returned %d document(s)", len(documents))

    return documents


def bm25_retrieve(query: str):
    """Retrieve documents using BM25 lexical search."""

    logger.info("Running BM25 retrieval | top_k=%d", BM25_TOP_K)

    bm25, documents = get_bm25_index()

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:BM25_TOP_K]

    results = [documents[index] for index in ranked_indices]

    logger.info("BM25 retrieval returned %d document(s)", len(results))

    return results


def reciprocal_rank_fusion(result_lists, k=RRF_K):
    """Combine ranked retrieval results using Reciprocal Rank Fusion."""

    logger.info("Applying Reciprocal Rank Fusion | rrf_k=%d", k)

    scores = {}
    documents = {}

    for result_list in result_lists:
        for rank, document in enumerate(result_list, start=1):
            document_id = document.metadata.get("chunk_id", document.page_content)

            if document_id not in scores:
                scores[document_id] = 0.0
                documents[document_id] = document

            scores[document_id] += 1 / (k + rank)

    ranked_documents = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    results = [documents[document_id] for document_id, _ in ranked_documents]

    logger.info("RRF produced %d unique document(s)", len(results))

    return results


def retrieve_documents(query: str, k: int = FINAL_TOP_K):
    """Retrieve documents using dense and sparse retrieval with RRF fusion."""

    logger.info("========== HYBRID RETRIEVAL STARTED ==========")
    logger.debug("Query received: %s", query)

    dense_documents = dense_retrieve(query)
    bm25_documents = bm25_retrieve(query)

    fused_documents = reciprocal_rank_fusion([dense_documents, bm25_documents])

    final_documents = fused_documents[:k]

    logger.info("Final hybrid retrieval returned %d document(s)", len(final_documents))
    logger.info("========== HYBRID RETRIEVAL COMPLETED ==========")

    return final_documents


if __name__ == "__main__":
    query = input("Enter your query: ")

    documents = retrieve_documents(query)

    print("\nRetrieved Hybrid Documents:\n")

    for index, document in enumerate(documents, start=1):
        print(f"--- Hybrid Document {index} ---")
        print(document.page_content)
        print("Metadata:", document.metadata)
        print()