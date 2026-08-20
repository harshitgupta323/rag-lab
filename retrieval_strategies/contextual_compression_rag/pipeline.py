import logging
from ingest import ingest_documents
from retrieve import create_retriever, retrieve_documents
from compress import create_compressor, compress_documents
from generate import create_llm, generate_answer

logger = logging.getLogger(__name__)


def run_pipeline(query):
    logger.info("Starting Contextual Compression RAG pipeline")

    ingest_documents()

    retriever = create_retriever()
    compressor = create_compressor()
    llm = create_llm()

    retrieved_documents = retrieve_documents(retriever, query)
    compressed_documents = compress_documents(compressor, query, retrieved_documents)
    answer = generate_answer(llm, query, compressed_documents)

    logger.info("Contextual Compression RAG pipeline completed successfully")

    return answer