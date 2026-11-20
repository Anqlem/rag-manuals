"""Rerankers: score (query, passage) pairs jointly, which bi-encoder vectors cannot do."""

from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

from rag.config import get_settings


class CohereReranker:
    def __init__(self, model: str):
        import cohere

        self.client, self.model = cohere.ClientV2(), model

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def scores(self, query: str, passages: list[str]) -> list[float]:
        resp = self.client.rerank(model=self.model, query=query, documents=passages)
        out = [0.0] * len(passages)
        for r in resp.results:
            out[r.index] = r.relevance_score
        return out


class LocalReranker:
    """pip install sentence-transformers. Runs on CPU; ~40 pairs take well under a second."""

    def __init__(self, model: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [float(x) for x in self.model.predict([(query, p) for p in passages])]


@lru_cache
def get_reranker():
    s = get_settings()
    if s.reranker == "cohere":
        return CohereReranker(s.cohere_rerank_model)
    if s.reranker == "local":
        return LocalReranker(s.local_rerank_model)
    return None
