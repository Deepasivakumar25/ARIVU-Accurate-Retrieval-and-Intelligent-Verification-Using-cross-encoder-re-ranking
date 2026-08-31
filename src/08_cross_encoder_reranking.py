import numpy as np
from sentence_transformers import CrossEncoder


def rerank(question: str, candidate_chunks: list[str], top_k: int = 3):
    context_list = [[question, chunk] for chunk in candidate_chunks]
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = cross_encoder_model.predict(context_list)
    top_score_indices = np.argsort(scores)[::-1]
    best_chunks = [candidate_chunks[idx] for idx in top_score_indices[:top_k]]
    return best_chunks, scores, top_score_indices
