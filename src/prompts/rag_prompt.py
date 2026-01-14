def build_rag_prompt(context: str, query: str) -> str:
    return f"""
You are a precise assistant.

Use ONLY the information provided in the context below.
If the answer cannot be derived from the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
""".strip()
