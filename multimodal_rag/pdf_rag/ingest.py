import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import DATA_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_documents():
    documents = []

    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))

    logger.info("Found %d PDF files", len(pdf_files))

    for pdf_file in pdf_files:
        logger.info("Loading PDF: %s", pdf_file.name)

        loader = PyPDFLoader(str(pdf_file))
        pdf_documents = loader.load()

        documents.extend(pdf_documents)

        logger.info("Loaded %d pages from %s", len(pdf_documents), pdf_file.name)

    logger.info("Total PDF pages loaded: %d", len(documents))

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
        raise ValueError(f"No PDF files found in {DATA_DIR}")

    chunks = split_documents(documents)

    return create_vector_store(chunks)