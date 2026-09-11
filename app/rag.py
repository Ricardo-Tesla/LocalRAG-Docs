import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from ollama import Client
from app.ingestion import load_and_chunk_pdf
from pathlib import Path
import uuid
import os
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords

import re

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, remove stopwords."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return [word for word in text.split() if word not in STOPWORDS]


try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))
    
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

# In-memory BM25 index, rebuilt whenever ingestion adds new chunks.
# None until first built; simple invalidate-and-rebuild rather than
# incremental updates, since our chunk counts are small enough that
# rebuilding from scratch is fast and far simpler to reason about.
_bm25_index = None
_bm25_chunk_lookup = None  # parallel list: chunk metadata matching bm25 index order


def _build_bm25_index():
    global _bm25_index, _bm25_chunk_lookup

    all_chunks = collection.get(include=["documents", "metadatas"])
    texts = all_chunks["documents"]
    metadatas = all_chunks["metadatas"]

    tokenized = [_tokenize(text) for text in texts]
    
    _bm25_index = BM25Okapi(tokenized)
    _bm25_chunk_lookup = [
        {"text": text, "page": meta["page"], "source_file": meta["source_file"]}
        for text, meta in zip(texts, metadatas)
    ]

def _bm25_search(query: str, n_results: int = 5) -> list[dict]:
    """
    Score all chunks against the query using BM25 (keyword/term-frequency
    matching), returning the top n_results with a normalized 0-1 score.

    BM25 scores are unbounded (not naturally 0-1 like cosine similarity),
    so we min-max normalize across this query's results to make them
    comparable to vector similarity scores when merging.
    """
    if _bm25_index is None:
        _build_bm25_index()

    if not _bm25_chunk_lookup:  # no documents ingested yet
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    max_score = max(scores) if len(scores) > 0 and max(scores) > 0 else 1.0

    scored_chunks = [
        {**_bm25_chunk_lookup[i], "bm25_score": round(scores[i] / max_score, 3)}
        for i in range(len(scores))
        if scores[i] > 0  # exclude chunks with zero keyword overlap entirely
    ]

    scored_chunks.sort(key=lambda c: c["bm25_score"], reverse=True)
    return scored_chunks[:n_results]

def retrieve_hybrid(query: str, n_results: int = 5, rrf_k: int = 60) -> list[dict]:
    """
    Combine vector similarity search and BM25 keyword search using
    Reciprocal Rank Fusion (RRF) rather than raw score averaging.

    An earlier version combined raw normalized scores directly, but with
    a small corpus (~100 chunks), per-query max-normalization proved
    unstable: a single coincidental keyword match could be the top result
    in a weak field, then get inflated to a score of 1.0, outranking
    genuinely relevant results. RRF avoids this entirely by combining
    RANK POSITIONS instead of raw scores, so no single method's score
    scale can dominate or distort the merge.
    """
    vector_results = retrieve(query, n_results=n_results * 3, min_similarity=0.0)
    bm25_results = _bm25_search(query, n_results=n_results * 3)

    rrf_scores = {}
    chunk_info = {}

    for rank, r in enumerate(vector_results, start=1):
        key = (r["source_file"], r["page"], r["text"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank)
        chunk_info[key] = r

    for rank, r in enumerate(bm25_results, start=1):
        key = (r["source_file"], r["page"], r["text"])
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rrf_k + rank)
        chunk_info.setdefault(key, r)

    results = []
    for key, score in rrf_scores.items():
        info = chunk_info[key]
        results.append({
            "text": info["text"],
            "page": info["page"],
            "source_file": info["source_file"],
            "similarity_score": round(score, 4),
        })

    results.sort(key=lambda s: s["similarity_score"], reverse=True)
    return results[:n_results]

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

    global _bm25_index
    _bm25_index = None  # invalidate cache; rebuilt lazily on next retrieval

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
        
        
    print("\n--- BM25 keyword search ---")
    bm25_results = _bm25_search("Smart Market Matchmaker")
    for r in bm25_results:
        print(f"  {r['source_file']} p{r['page']} — bm25_score={r['bm25_score']}")
        
    print("\n--- Hybrid retrieve ---")
    start = time.time()
    hybrid = retrieve_hybrid("What are the use cases proposed?")
    print(f"Took {time.time() - start:.1f}s, {len(hybrid)} sources")
    for s in hybrid:
        print(f"  {s['source_file']} p{s['page']} — {s['similarity_score']}")
        
    print("\n--- DEBUG: raw candidates feeding the merge ---")
    debug_question = "What are the use cases proposed?"

    print("Vector candidates (top 15, no threshold):")
    for rank, r in enumerate(retrieve(debug_question, n_results=15, min_similarity=0.0), start=1):
        print(f"  rank {rank}: {r['source_file']} p{r['page']} — score={r['similarity_score']}")

    print("\nBM25 candidates (top 15):")
    for rank, r in enumerate(_bm25_search(debug_question, n_results=15), start=1):
        print(f"  rank {rank}: {r['source_file']} p{r['page']} — bm25_score={r['bm25_score']}")