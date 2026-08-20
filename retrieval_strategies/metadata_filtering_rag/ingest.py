import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import DATA_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_documents():
    documents = []

    logger.info("Loading documents from %s", DATA_DIR)

    for file_path in Path(DATA_DIR).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        metadata = {
            "source": file_path.name,
            "category": file_path.stem.split("_")[0],
            "year": int(file_path.stem.split("_")[1])
        }

        documents.append(Document(page_content=text, metadata=metadata))

        logger.info("Loaded document: %s with metadata: %s", file_path.name, metadata)

    logger.info("Total documents loaded: %d", len(documents))

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    logger.info("Created %d document chunks", len(chunks))

    return chunks


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    if chunks:
        vector_store.add_documents(chunks)

    logger.info("Added %d chunks to Chroma", len(chunks))

    return vector_store


def ingest_documents():
    documents = load_documents()

    if not documents:
        raise ValueError(f"No documents found in {DATA_DIR}")

    chunks = split_documents(documents)

    return create_vector_store(chunks)