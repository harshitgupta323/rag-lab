import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
from sentence_transformers import CrossEncoder
from retrieval_strategies.reranking_rag.config import RERANKER_MODEL, RERANKED_TOP_K

logger = logging.getLogger(__name__)


class DocumentReranker:
    def __init__(self):
        logger.info("Loading reranker model: %s", RERANKER_MODEL)
        self.model = CrossEncoder(RERANKER_MODEL)
        logger.info("Reranker initialized successfully")

    def rerank(self, query, documents):
        if not documents:
            logger.warning("No documents available for reranking")
            return []

        pairs = [(query, document.page_content) for document in documents]
        scores = self.model.predict(pairs)

        ranked_documents = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
        top_documents = ranked_documents[:RERANKED_TOP_K]

        logger.info("Reranked %d documents and selected top %d", len(documents), len(top_documents))

        reranked_documents = []

        for rank, (document, score) in enumerate(top_documents, start=1):
            document.metadata["rerank_score"] = float(score)
            document.metadata["rerank_rank"] = rank
            reranked_documents.append(document)
            logger.info("Rank %d | score=%.4f | source=%s", rank, score, document.metadata.get("source", "unknown"))

        return reranked_documents