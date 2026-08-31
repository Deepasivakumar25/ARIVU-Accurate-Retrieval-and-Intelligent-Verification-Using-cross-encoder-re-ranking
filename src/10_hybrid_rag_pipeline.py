from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from _03_pdf_extraction import extract_pdf_text
from _04_text_chunking import create_chunks
from _05_llm_setup import load_chatbot
from _06_semantic_search import build_semantic_index, semantic_search
from _07_keyword_search import keyword_search, combine_indices
from _08_cross_encoder_reranking import rerank
from _09_rag_generation import generate_answer


def run_pipeline(pdf_path: str, question: str) -> str:
    pdf_text = extract_pdf_text(pdf_path)
    chunks = create_chunks(pdf_text)
    embedding_model, index = build_semantic_index(chunks)
    _, semantic_indices = semantic_search(question, embedding_model, index, chunks)
    _, keyword_indices = keyword_search(chunks, question)
    unique_indices = combine_indices(keyword_indices, semantic_indices)
    candidates = [chunks[i] for i in unique_indices if 0 <= i < len(chunks)]
    best_chunks, _, _ = rerank(question, candidates)
    context = "\n\n".join(best_chunks)
    chatbot = load_chatbot()
    return generate_answer(chatbot, question, context)
