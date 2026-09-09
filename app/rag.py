import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from app.ingestion import load_and_chunk_pdf
import os
from pathlib import Path
from ollama import Client
import uuid

ollama_client = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))


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


def ingest_pdf(pdf_path: str) -> dict:
    """Load, chunk, embed, and store a PDF in the vector store.

    Skips ingestion if this exact filename has already been ingested,
    preventing duplicate chunks from silently degrading retrieval quality
    (a real issue we hit: re-ingesting the same file produced duplicate,
    unmerged chunks that skewed similarity rankings).
    """
    filename = Path(pdf_path).name

    existing = collection.get(where={"source_file": filename}, include=[])
    if existing["ids"]:
        print(f"Skipped: '{filename}' is already ingested ({len(existing['ids'])} existing chunks).")
        return {"status": "skipped", "reason": "already_ingested", "filename": filename}

    chunks = load_and_chunk_pdf(pdf_path)
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "source_file": c["source_file"]} for c in chunks]
    ids = [f"{uuid.uuid4()}_{i}" for i in range(len(texts))]

    embeddings = embedding_model.encode(texts).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"Ingested {len(texts)} chunks from {pdf_path}")
    return {"status": "ingested", "chunk_count": len(texts), "filename": filename}


def retrieve(query: str, n_results: int = 5, min_similarity: float = 0.2):
    """Embed the query and fetch the most similar chunks, with metadata + scores.

    Chunks below min_similarity are discarded — Chroma always returns the
    closest N vectors regardless of how weak the match is, so this filters
    out results that aren't actually relevant rather than presenting them
    as if they were.
    """
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
        similarity_score = round(1 - distance, 3)
        if similarity_score < min_similarity:
            continue

        sources.append({
            "text": text,
            "page": metadata["page"],
            "source_file": metadata["source_file"],
            "similarity_score": similarity_score,
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
    """
    Full RAG pipeline: retrieve -> build prompt -> generate -> return answer + sources.

    Uses multi-query retrieval by default (v2) rather than a single embedding
    search — this trades ~25s of added latency for meaningfully better
    retrieval on ambiguously or awkwardly phrased questions (see README,
    Version 2: Upgrade Rationale).
    """
    sources = retrieve_multi_query(query)

    if not sources:
        return {
            "answer": "I don't have enough information in the uploaded documents to answer that.",
            "sources": [],
        }

    prompt = build_prompt(query, sources)

    response = ollama_client.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response["message"]["content"],
        "sources": sources,
    }

def generate_query_variants(question: str, n_variants: int = 2) -> list[str]:
    """
    Ask the LLM for alternative phrasings of the question, to compensate for
    retrieval's sensitivity to exact wording (see Known Limitations, v1).
    Returns the original question plus up to n_variants alternatives.
    """
    prompt = f"""Generate {n_variants} alternative phrasings of the following question.
The alternatives should ask for the same information using different words.
Return ONLY the alternative questions, one per line, with no numbering or extra text.

Original question: {question}"""

    response = ollama_client.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
    )

    variants = [
        line.strip()
        for line in response["message"]["content"].strip().split("\n")
        if line.strip()
    ]

    return [question] + variants[:n_variants]

def retrieve_multi_query(question: str, n_results: int = 5, min_similarity: float = 0.2) -> list[dict]:
    """
    Retrieve using the original question plus LLM-generated rephrasings,
    merging results and keeping each chunk's best score across all variants.
    """
    variants = generate_query_variants(question)

    best_by_id = {}
    for variant in variants:
        results = retrieve(variant, n_results=n_results, min_similarity=min_similarity)
        for source in results:
            key = (source["source_file"], source["page"], source["text"])
            if key not in best_by_id or source["similarity_score"] > best_by_id[key]["similarity_score"]:
                best_by_id[key] = source

    merged = sorted(best_by_id.values(), key=lambda s: s["similarity_score"], reverse=True)
    return merged[:n_results]

if __name__ == "__main__":
    import time

    question = "What is the first use case according to the document uploaded?"

    print("--- Single-query retrieve (baseline) ---")
    start = time.time()
    baseline = retrieve(question)
    print(f"Took {time.time() - start:.1f}s, {len(baseline)} sources")
    for s in baseline:
        print(f"  {s['source_file']} p{s['page']} — {s['similarity_score']}")

    print("\n--- Multi-query retrieve ---")
    start = time.time()
    multi = retrieve_multi_query(question)
    print(f"Took {time.time() - start:.1f}s, {len(multi)} sources")
    for s in multi:
        print(f"  {s['source_file']} p{s['page']} — {s['similarity_score']}")