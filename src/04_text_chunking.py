def create_chunks(pdf_text: str, chunk_size: int = 50) -> list[str]:
    word_split = pdf_text.split()
    return [
        " ".join(word_split[i:i + chunk_size])
        for i in range(0, len(word_split), chunk_size)
    ]
