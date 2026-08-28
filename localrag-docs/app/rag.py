import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# --- Setup (runs once when this module is imported) ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="documents")


def retrieve(query: str, n_results: int = 3):
    """Embed the query and fetch the most similar chunks from Chroma."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    return results["documents"][0]  # list of chunk texts


def build_prompt(query: str, chunks: list[str]) -> str:
    """Combine retrieved chunks + question into a grounded prompt."""
    context = "\n\n---\n\n".join(chunks)

    prompt = f"""You are a helpful assistant answering questions based ONLY on the context below.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Do not use any outside knowledge.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(query: str) -> str:
    """Full RAG pipeline: retrieve -> build prompt -> generate."""
    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    question = "What are the key responsibilities?"
    answer = generate_answer(question)
    print("Question:", question)
    print("\nAnswer:", answer)