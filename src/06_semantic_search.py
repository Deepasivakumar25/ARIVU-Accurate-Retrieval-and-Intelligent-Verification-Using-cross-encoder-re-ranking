from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def build_semantic_index(chunk_list: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer(model_name)
    chunk_embedding = embedding_model.encode(chunk_list)
    dimension = chunk_embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embedding)
    return embedding_model, index


def semantic_search(question: str, embedding_model, index, chunk_list: list[str], top_k: int = 5):
    question_embedding = embedding_model.encode([question])
    distance, index_number = index.search(np.array(question_embedding), k=top_k)
    indices = [int(i) for i in index_number[0] if 0 <= i < len(chunk_list)]
    return [chunk_list[i] for i in indices], indices
