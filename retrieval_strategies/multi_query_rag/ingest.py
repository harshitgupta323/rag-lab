import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from shared.utils.logging_config import setup_logging, get_logger

from config import DATA_DIR, VECTOR_DB_DIR, LOG_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def load_documents():
    """Load text documents from the data directory."""

    logger.info("Loading documents from: %s", DATA_DIR)

    documents = []
    files = list(DATA_DIR.glob("*.txt"))

    logger.info("Found %d text file(s)", len(files))

    for file_path in files:
        logger.info("Loading document: %s", file_path.name)
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    logger.info("Successfully loaded %d document(s)", len(documents))

    return documents


def split_documents(documents):
    """Split documents into chunks."""

    logger.info("Splitting documents | chunk_size=%d | chunk_overlap=%d", CHUNK_SIZE, CHUNK_OVERLAP)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    logger.info("Created %d document chunks", len(chunks))

    return chunks


def create_vector_store(chunks):
    """Create the Chroma vector store."""

    logger.info("Initializing embedding model: %s", EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("Creating Chroma vector store")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name="multi_query_rag",
    )

    logger.info("Chroma vector store created successfully | path=%s", VECTOR_DB_DIR)

    return vector_store


def ingest():
    """Run the complete ingestion pipeline."""

    logger.info("========== MULTI-QUERY RAG INGESTION STARTED ==========")

    try:
        documents = load_documents()

        if not documents:
            logger.error("No .txt documents found in %s", DATA_DIR)
            raise ValueError(f"No .txt documents found in {DATA_DIR}")

        chunks = split_documents(documents)
        create_vector_store(chunks)

        logger.info("========== MULTI-QUERY RAG INGESTION COMPLETED ==========")

    except Exception:
        logger.exception("Multi-Query RAG ingestion failed")
        raise


if __name__ == "__main__":
    ingest()