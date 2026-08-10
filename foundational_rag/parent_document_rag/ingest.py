import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever

from shared.utils.logging_config import setup_logging, get_logger

from config import DATA_DIR, VECTOR_DB_DIR, LOG_DIR, EMBEDDING_MODEL, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP


setup_logging(LOG_DIR)
logger = get_logger(__name__)


def load_documents():
    """Load documents from the data directory."""

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


def create_retriever():
    """Create the ParentDocumentRetriever."""

    logger.info("Initializing embedding model: %s", EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        collection_name="parent_document_rag",
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings,
    )

    parent_store = InMemoryStore()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)

    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=parent_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        child_metadata_fields=["source"],
    )

    logger.info("ParentDocumentRetriever created successfully")
    logger.info("Parent chunk size=%d | child chunk size=%d", PARENT_CHUNK_SIZE, CHILD_CHUNK_SIZE)

    return retriever


def ingest():
    """Ingest documents into the parent-child retrieval system."""

    logger.info("========== PARENT DOCUMENT RAG INGESTION STARTED ==========")

    try:
        documents = load_documents()

        if not documents:
            logger.error("No .txt documents found in %s", DATA_DIR)
            raise ValueError(f"No .txt documents found in {DATA_DIR}")

        retriever = create_retriever()

        logger.info("Adding documents to ParentDocumentRetriever")

        retriever.add_documents(documents)

        logger.info("========== PARENT DOCUMENT RAG INGESTION COMPLETED ==========")

        return retriever

    except Exception:
        logger.exception("Parent Document RAG ingestion failed")
        raise


if __name__ == "__main__":
    ingest()