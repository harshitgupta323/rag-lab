import logging
import os
from pathlib import Path

from PIL import Image
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from config import DATA_DIR, CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, TEMPERATURE

logger = logging.getLogger(__name__)


def create_vision_llm():
    logger.info("Initializing vision model: %s", LLM_MODEL)

    return ChatGroq(model=LLM_MODEL, temperature=TEMPERATURE, api_key=os.getenv("GROQ_API_KEY"))


def load_images():
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [file_path for file_path in Path(DATA_DIR).iterdir() if file_path.suffix.lower() in image_extensions]

    logger.info("Found %d images", len(image_files))

    return image_files


def generate_image_description(llm, image_path):
    logger.info("Generating description for image: %s", image_path.name)

    image = Image.open(image_path)

    prompt = """Describe this image in detail for use in a visual search system.

    Include:
    - Main objects
    - People, if present
    - Actions
    - Scene or environment
    - Important visual attributes
    - Text visible in the image
    - Relationships between important objects

    Return a concise but information-rich description."""

    response = llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower()};base64,{_encode_image(image_path)}"}}
            ]
        }
    ])

    description = response.content.strip()

    logger.info("Generated image description for %s", image_path.name)

    return description


def _encode_image(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    if documents:
        vector_store.add_documents(documents)

    logger.info("Added %d image descriptions to Chroma", len(documents))

    return vector_store


def ingest_images():
    image_files = load_images()

    if not image_files:
        raise ValueError(f"No images found in {DATA_DIR}")

    llm = create_vision_llm()
    documents = []

    for image_path in image_files:
        description = generate_image_description(llm, image_path)

        documents.append(
            Document(
                page_content=description,
                metadata={
                    "source": image_path.name,
                    "path": str(image_path)
                }
            )
        )

    return create_vector_store(documents)