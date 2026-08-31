import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def keyword_search(chunk_list: list[str], question: str, top_k: int = 5):
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(chunk_list)
    question_tfidf = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_tfidf, x)
    top_indices = np.argsort(similarity_scores[0])[::-1][:top_k]
    keyword_chunks = [chunk_list[rec] for rec in top_indices]
    return keyword_chunks, top_indices.tolist()


def combine_indices(keyword_indices, semantic_indices):
    return list(dict.fromkeys(keyword_indices + semantic_indices))
