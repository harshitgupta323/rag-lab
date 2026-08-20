import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from retrieval_strategies.reranking_rag.config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


def create_retriever():
    logger.info("Initializing embedding model: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name="reranking_rag", embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    logger.info("Retriever initialized with top_k=%d", RETRIEVAL_TOP_K)

    return retriever


def retrieve_documents(retriever, query):
    logger.info("Retrieving candidate documents for query: %s", query)

    documents = retriever.invoke(query)

    logger.info("Retrieved %d candidate documents", len(documents))
    return documents