import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.utils.logging_config import setup_logging, get_logger

from config import VECTOR_DB_DIR, LOG_DIR, EMBEDDING_MODEL, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP, TOP_K


setup_logging(LOG_DIR)
logger = get_logger(__name__)


# def get_retriever():
#     """Load the ParentDocumentRetriever."""

#     logger.info("Loading ParentDocumentRetriever")

#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

#     vector_store = Chroma(
#         collection_name="parent_document_rag",
#         persist_directory=str(VECTOR_DB_DIR),
#         embedding_function=embeddings,
#     )

#     parent_store = InMemoryStore()

#     parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)

#     child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)

#     retriever = ParentDocumentRetriever(
#         vectorstore=vector_store,
#         docstore=parent_store,
#         child_splitter=child_splitter,
#         parent_splitter=parent_splitter,
#         child_metadata_fields=["source"],
#     )

#     logger.info("ParentDocumentRetriever loaded successfully")

#     return retriever


def retrieve_documents(retriever, query: str, k: int = TOP_K):
    """Retrieve parent documents using child chunk similarity."""

    logger.info("Retrieving parent documents | top_k=%d", k)
    logger.debug("Query received: %s", query)

    # retriever = get_retriever()
    retriever.search_kwargs = {"k": k}

    documents = retriever.invoke(query)

    logger.info("Retrieved %d parent document(s)", len(documents))

    return documents


if __name__ == "__main__":
    query = input("Enter your query: ")
    documents = retrieve_documents(query)

    print("\nRetrieved Parent Documents:\n")

    for index, document in enumerate(documents, start=1):
        print(f"--- Parent Document {index} ---")
        print(document.page_content)
        print("Metadata:", document.metadata)
        print()