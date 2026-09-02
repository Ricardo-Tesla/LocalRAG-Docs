import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from app.ingestion import load_and_chunk_pdf

# all-MiniLM-L6-v2: small (~80MB), CPU-friendly, 384-dim. Not the strongest
# embedding model available, but the standard local baseline — swappable
# later without touching the rest of the pipeline.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

# hnsw:space="cosine": Chroma defaults to L2 (Euclidean) distance, which
# doesn't produce a meaningful 0-1 similarity score for text embeddings.
# Cosine is required for the "similarity = 1 - distance" conversion below
# to be valid. This setting only takes effect at collection creation —
# changing it later means deleting and rebuilding the store.
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


def ingest_pdf(pdf_path: str):
    """Load, chunk, embed, and store a PDF in the vector store."""
    chunks = load_and_chunk_pdf(pdf_path)
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "source_file": c["source_file"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(texts))]

    embeddings = embedding_model.encode(texts).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Ingested {len(texts)} chunks from {pdf_path}")


def retrieve(query: str, n_results: int = 5):
    """Embed the query and fetch the most similar chunks, with metadata + scores."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    sources = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        sources.append({
            "text": text,
            "page": metadata["page"],
            "source_file": metadata["source_file"],
            # Chroma returns distance (lower = closer); this flips it to an
            # intuitive similarity score (higher = more relevant).
            "similarity_score": round(1 - distance, 3),
        })
    return sources


def build_prompt(query: str, sources: list[dict]) -> str:
    """Combine retrieved chunks + question into a grounded prompt."""
    context = "\n\n---\n\n".join(
        f"[Page {s['page']}] {s['text']}" for s in sources
    )

    # Explicitly instructed to refuse rather than guess when context is
    # insufficient — this is what makes the system "grounded" instead of
    # a general-purpose chatbot with extra text pasted in.
    prompt = f"""You are a helpful assistant answering questions based ONLY on the context below.
If the answer is not contained in the context, say "I don't have enough information to answer that."
Do not use any outside knowledge.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(query: str) -> dict:
    """Full RAG pipeline: retrieve -> build prompt -> generate -> return answer + sources."""
    sources = retrieve(query)
    prompt = build_prompt(query, sources)

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources,
    }


if __name__ == "__main__":
    # Manual entry point for local testing outside the API — the FastAPI
    # /upload and /query endpoints call these same functions directly.
    ingest_pdf("data/uploads/sample.pdf")

    question = "What are the key responsibilities?"
    result = generate_answer(question)

    print("Question:", question)
    print("\nAnswer:", result["answer"])
    print("\n--- Sources ---")
    for s in result["sources"]:
        print(f"Page {s['page']} (score: {s['similarity_score']}): {s['text'][:80]}...")