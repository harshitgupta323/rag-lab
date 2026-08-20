import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from retrieval_strategies.reranking_rag.config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_documents():
    documents = []
    logger.info("Loading documents from %s", DATA_DIR)

    for file_path in Path(DATA_DIR).glob("*.txt"):
        try:
            text = file_path.read_text(encoding="utf-8")
            documents.append(Document(page_content=text, metadata={"source": file_path.name}))
            logger.info("Loaded document: %s", file_path.name)
        except Exception:
            logger.exception("Failed to load document: %s", file_path)

    logger.info("Loaded %d documents", len(documents))
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    logger.info("Created %d document chunks", len(chunks))
    return chunks


def create_vector_store(chunks):
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("Creating Chroma vector store")
    vector_store = Chroma(collection_name="reranking_rag", embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    if chunks:
        vector_store.add_documents(chunks)

    logger.info("Stored %d chunks in Chroma", len(chunks))
    return vector_store


def ingest_documents():
    documents = load_documents()

    if not documents:
        raise ValueError("No documents found in the data directory.")

    chunks = split_documents(documents)
    return create_vector_store(chunks)