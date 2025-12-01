# app/services/reranker_service.py

from sentence_transformers import CrossEncoder

# Load once, keep in memory
reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank(query, docs):
    """
    docs: list of langchain Document objects
    returns: docs sorted by reranker score (descending)
    """
    if not docs:
        return []

    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    # attach scores
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    # return only sorted docs
    return [doc for doc, score in ranked]
