import base64
import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, DOCUMENTS_DIR, IMAGES_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE

logger = logging.getLogger(__name__)


def create_vision_llm():
    logger.info("Initializing vision LLM: %s", LLM_MODEL)

    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE)


def load_pdf_documents():
    documents = []

    pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))

    logger.info("Found %d PDF files", len(pdf_files))

    for pdf_file in pdf_files:
        logger.info("Loading PDF: %s", pdf_file.name)

        loader = PyPDFLoader(str(pdf_file))
        pdf_documents = loader.load()

        for document in pdf_documents:
            document.metadata["modality"] = "text"
            document.metadata["source_type"] = "pdf"
            document.metadata["source"] = pdf_file.name

        documents.extend(pdf_documents)

        logger.info("Loaded %d pages from %s", len(pdf_documents), pdf_file.name)

    return documents


def split_pdf_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    logger.info("Created %d PDF text chunks", len(chunks))

    return chunks


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_image_description(llm, image_path):
    logger.info("Generating description for image: %s", image_path.name)

    image_data = encode_image(image_path)

    extension = image_path.suffix.lower().replace(".", "")

    if extension == "jpg":
        extension = "jpeg"

    prompt = """Describe this image for a multimodal retrieval system.

    Include:
    - Main objects
    - People
    - Actions
    - Environment
    - Important visual attributes
    - Visible text
    - Relationships between important objects

    Return a concise but information-rich description."""

    response = llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{extension};base64,{image_data}"}}
            ]
        }
    ])

    description = response.content.strip()

    logger.info("Generated description for %s", image_path.name)

    return description


def load_image_documents(llm):
    documents = []

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [file_path for file_path in Path(IMAGES_DIR).iterdir() if file_path.suffix.lower() in image_extensions]

    logger.info("Found %d image files", len(image_files))

    for image_path in image_files:
        description = generate_image_description(llm, image_path)

        documents.append(
            Document(
                page_content=description,
                metadata={
                    "source": image_path.name,
                    "path": str(image_path),
                    "modality": "image",
                    "source_type": "image"
                }
            )
        )

    return documents


def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    if documents:
        vector_store.add_documents(documents)

    logger.info("Added %d multimodal documents to Chroma", len(documents))

    return vector_store


def ingest_documents():
    llm = create_vision_llm()

    pdf_documents = load_pdf_documents()
    pdf_chunks = split_pdf_documents(pdf_documents) if pdf_documents else []

    image_documents = load_image_documents(llm)

    documents = pdf_chunks + image_documents

    if not documents:
        raise ValueError(f"No PDF or image files found in {DATA_DIR}")

    logger.info("Total multimodal documents prepared: %d", len(documents))

    return create_vector_store(documents)