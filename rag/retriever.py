import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

from rag.embedder import load_index, MODEL_NAME

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3


def search(query: str, index: faiss.Index, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            chunk = chunks[idx].copy()
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

    return results


def ask(query: str, index: faiss.Index, chunks: list[dict]) -> dict:
    relevant_chunks = search(query, index, chunks)

    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        context_parts.append(
            f"[Source {i+1} — {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant answering questions based on the provided document excerpts.

Use only the information in the context below to answer the question.
Always cite which source and page your answer comes from.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    message = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = message.choices[0].message.content

    return {
        "question": query,
        "answer": answer,
        "sources": relevant_chunks
    }


if __name__ == "__main__":
    print("Loading index...")
    index, chunks = load_index()

    test_questions = [
        "What is the attention mechanism?",
        "What optimizer was used to train the model?",
        "How many layers does the encoder have?"
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print('='*60)
        result = ask(question, index, chunks)
        print(f"A: {result['answer']}")
        print(f"\nSources used:")
        for s in result['sources']:
            print(f"  - Page {s['page']} (score: {s['score']}): {s['text'][:80]}...")