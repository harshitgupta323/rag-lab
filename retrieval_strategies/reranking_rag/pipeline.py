import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from retrieval_strategies.reranking_rag.ingest import ingest_documents
from retrieval_strategies.reranking_rag.retrieve import create_retriever, retrieve_documents
from retrieval_strategies.reranking_rag.rerank import DocumentReranker
from retrieval_strategies.reranking_rag.generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def build_pipeline():
    logger.info("Building Reranking RAG pipeline")

    ingest_documents()

    retriever = create_retriever()
    reranker = DocumentReranker()
    llm = create_llm()

    logger.info("Reranking RAG pipeline initialized successfully")

    return retriever, reranker, llm


def run_pipeline(query, retriever, reranker, llm):
    logger.info("Running Reranking RAG pipeline")

    retrieved_documents = retrieve_documents(retriever, query)
    reranked_documents = reranker.rerank(query, retrieved_documents)
    answer = generate_answer(llm, query, reranked_documents)

    logger.info("Reranking RAG pipeline completed successfully")

    return answer, retrieved_documents, reranked_documents