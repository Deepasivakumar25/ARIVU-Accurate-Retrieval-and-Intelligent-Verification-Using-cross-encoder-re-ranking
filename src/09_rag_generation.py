def generate_answer(chatbot, question: str, context: str, max_new_tokens: int = 120) -> str:
    prompt = f"""<|user|>
Use ONLY the context below.

Context:
{context}

Question:
{question}

If the answer is not present, reply exactly:
I couldn't find that information.

<|assistant|>
"""
    response = chatbot(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return response[0]["generated_text"].strip()
