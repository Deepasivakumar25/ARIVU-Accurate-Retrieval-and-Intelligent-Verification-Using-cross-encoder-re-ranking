import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build_index(self, chunks: list[str]) -> None:
        self.chunks = chunks
        embeddings = self.model.encode(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.asarray(embeddings))

    def search(self, query: str, top_k: int = 5) -> tuple[list[str], list[int]]:
        query_embedding = self.model.encode([query])
        _, indices = self.index.search(np.asarray(query_embedding), top_k)
        valid_indices = [int(i) for i in indices[0] if 0 <= i < len(self.chunks)]
        return [self.chunks[i] for i in valid_indices], valid_indices
