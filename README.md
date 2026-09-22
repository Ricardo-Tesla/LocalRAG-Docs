# LocalRAG Docs

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system for technical document question answering. Users upload PDF documents, and the system retrieves relevant passages and generates grounded, source-attributed answers — entirely on-device, with no data sent to third-party APIs.

## Overview

LocalRAG Docs allows a user to upload one or more PDF documents and ask natural-language questions about their contents. The system retrieves the most relevant passages using an advanced, multi-stage retrieval pipeline, then passes those passages to a locally-hosted language model to generate an answer. Every answer is returned alongside the exact source passages, page numbers, and relevance scores used to produce it, allowing the user to verify the response against the original material.

The entire pipeline — document parsing, embedding generation, vector search, and language model inference — runs locally. No document content, queries, or generated answers leave the user's machine at any point.

## How to Use This App

**There is no live hosted demo, by design.** LocalRAG Docs is built to run entirely on your own machine, so that no document, question, or answer ever leaves your device — that privacy guarantee is the core point of the project, not an afterthought. This means there's no link to click and try it instantly in a browser; instead, anyone wanting to use it runs it locally, the same way you'd run a tool like Ollama or a self-hosted app such as Jellyfin or Home Assistant.

For a quick look at the app in action without installing anything, see the demo video linked from this repository or my portfolio. To actually run it yourself:

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

```bash
git clone <this-repository-url>
cd localrag-docs
docker compose up --build
```

Then, one time only, pull the language model into the running Ollama container:

```bash
docker exec -it localragdocs-ollama-1 ollama pull phi3
```

Once both steps finish, open `http://localhost:8501` in your browser. The app is now running entirely on your machine — upload a PDF, ask a question, and see grounded, source-attributed answers with no data sent anywhere else. Full setup details, including running without Docker, are in the Setup and Installation section below.

## Version 2: Upgrade Rationale

Version 1 established a complete, working local RAG pipeline: ingestion, chunking, embeddings, vector storage, grounded generation, and source attribution, wrapped in a FastAPI backend and Streamlit frontend, containerized with Docker.

Testing v1 against real, varied documents surfaced a specific, recurring weakness: retrieval quality was highly sensitive to how literally a question was phrased. A question like "What is the first use case according to the document uploaded?" returned largely irrelevant sources, while a more direct phrasing of the same underlying question scored substantially higher against the correct document. This is a known limitation of pure semantic (embedding-based) retrieval: it matches meaning and wording together, not intent alone.

Version 2 addresses this and related gaps through advanced retrieval and evaluation techniques, developed as targeted responses to specific weaknesses observed in v1. Every addition is held to three constraints:

- **Privacy** — no addition introduces a call to an external API. Query rephrasing and re-ranking both use local models, consistent with the project's fully-local design.
- **Performance** — since local, CPU-only LLM inference is already the dominant latency cost, every addition's latency impact is measured and stated explicitly.
- **Security** — file handling and input validation are held to the same standard as the original endpoints, extended to new ones.

### Advanced retrieval pipeline (`retrieve_advanced`)

Three techniques are combined into a single pipeline, each addressing a different, separately-diagnosed weakness:

1. **Multi-query generation** — the LLM generates 2–3 alternative phrasings of the user's question before retrieval, compensating for retrieval's sensitivity to exact wording.
2. **Hybrid search** — vector similarity (semantic) and BM25 (keyword/term-frequency) search are combined using Reciprocal Rank Fusion (RRF), catching exact-term matches (proper nouns, specific terminology) that pure semantic search under-ranks. An earlier version combined raw normalized scores directly; with a small corpus, per-query max-normalization proved unstable, since a single coincidental keyword match could be inflated to a top score and outrank genuinely relevant results. RRF combines rank positions instead, avoiding this failure mode entirely.
3. **Cross-encoder re-ranking** — the merged candidate pool is re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which processes the query and each chunk jointly rather than independently, producing a more accurate final relevance judgment. Results below a minimum re-rank score are discarded rather than padded into the response.

Each stage was validated in isolation against real test documents before being combined, and the combined pipeline was validated against the same test question that originally exposed the phrasing-sensitivity weakness — moving from a mix of irrelevant and relevant sources to a clean, sharply-separated result set.

## Motivation

Organizations in regulated or sensitive sectors (financial services, healthcare, industrial and manufacturing operations) frequently need to query internal technical documentation but cannot send proprietary or confidential content to third-party AI APIs. LocalRAG Docs demonstrates a production-oriented architecture for this exact requirement: a complete RAG pipeline with source transparency and advanced retrieval, built entirely on open-source, self-hostable components.

## Architecture

| Layer | Component | Responsibility |
|---|---|---|
| Frontend | Streamlit | Document upload, question input, answer and source display |
| Backend API | FastAPI | Exposes ingestion and query logic over HTTP (`/upload`, `/query`, `/health`) |
| Retrieval | Multi-query + hybrid search (BM25 + vector) + cross-encoder re-ranking | Retrieves and ranks relevant chunks for a given question |
| Generation | Ollama (phi3) | Generates grounded answers from retrieved context |
| Storage | ChromaDB | Persistent local vector store for document embeddings and metadata |

### Data flow

1. A user uploads a PDF through the Streamlit interface.
2. The file is sent to the FastAPI backend's `/upload` endpoint, hashed (to detect duplicate content regardless of filename), and saved to local disk.
3. The document is parsed page by page, split into overlapping text chunks, and each chunk is tagged with metadata (source filename, page number, content hash).
4. Each chunk is embedded and stored in a persistent ChromaDB collection; a parallel in-memory BM25 index is invalidated and rebuilt lazily on the next query.
5. When a user submits a question, the advanced retrieval pipeline runs: query variants are generated, hybrid search retrieves candidates for each variant, results are merged, and a cross-encoder re-ranks the merged pool down to a final top-k.
6. The final chunks are inserted into a structured, grounded prompt and sent to the local LLM via Ollama.
7. The generated answer is returned to the frontend along with the source chunks, their page numbers, originating filenames, and relevance scores.

## Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.12 |
| Document parsing | pypdf |
| Text chunking | LangChain (RecursiveCharacterTextSplitter) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Keyword search | rank-bm25, NLTK (stopword filtering) |
| Re-ranking | sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Vector database | ChromaDB (persistent, local) |
| LLM inference | Ollama (phi3) |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Data validation | Pydantic |
| Containerization | Docker, Docker Compose |

All components are free, open-source, and run without any external network calls at inference time.

## Project Structure

```
localrag-docs/
├── .streamlit/
│   └── config.toml        # UI theme configuration
├── app/
│   ├── ingestion.py       # PDF loading and chunking, with page/source metadata
│   ├── rag.py             # Ingestion, advanced retrieval, prompt construction, generation
│   ├── main.py            # FastAPI application (/health, /upload, /query)
│   ├── frontend.py        # Streamlit user interface
│   └── evaluate.py        # Automated retrieval regression checks
├── data/
│   └── uploads/           # Uploaded source documents
├── chroma_db/             # Persistent vector store (generated, not versioned)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup and Installation

### Option A: Docker Compose (recommended)

**Prerequisites:** Docker Desktop.

```bash
git clone <repository-url>
cd localrag-docs

docker compose up --build
```

This builds and starts three containers: `backend` (FastAPI), `frontend` (Streamlit), and `ollama` (LLM runtime). On first run, pull the language model into the containerized Ollama instance:

```bash
docker exec -it localragdocs-ollama-1 ollama pull phi3
```

The model persists in a named Docker volume, so this step is only needed once. The frontend is available at `http://localhost:8501`, the API at `http://localhost:8000`.

### Option B: Run directly on the host

**Prerequisites:** Python 3.12, [Ollama](https://ollama.com) installed locally.

```bash
git clone <repository-url>
cd localrag-docs

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
ollama pull phi3
```

Run the backend and frontend as two separate processes:

```bash
uvicorn app.main:app --reload
```

In a second terminal:

```bash
streamlit run app/frontend.py
```

### Running the evaluation script

```bash
python -m app.evaluate
```

## Design Notes

**Cosine similarity.** The vector store is explicitly configured to use cosine distance rather than Chroma's default (Euclidean/L2), the appropriate metric for comparing semantic similarity of text embeddings.

**Similarity threshold.** Chroma always returns the closest available vectors regardless of match quality. A minimum similarity threshold filters out weak matches before they reach either the language model or the user; the pipeline returns an explicit "not enough information" response when nothing clears it, without calling the LLM at all in that case.

**Globally unique, content-verified chunk identifiers.** Chunk IDs are prefixed with a UUID per ingestion call to guarantee uniqueness across documents (an earlier per-ingestion counter caused chunks from a new document to silently overwrite same-indexed chunks from a previous one). Ingestion also checks both exact filename and a SHA-256 content hash before storing new chunks: the same document was uploaded twice under two slightly different filenames during testing, doubling its representation in the corpus and visibly distorting BM25's term-rarity statistics. Filename-only checking didn't catch this; content hashing does, regardless of filename.

**Reciprocal Rank Fusion over raw score blending.** Hybrid search initially combined vector and BM25 scores as a weighted average after min-max normalization. With a small corpus, a single coincidental keyword match could become the top result in a weak field and get normalized to a score of 1.0, outranking genuinely relevant results. RRF combines rank positions instead of raw scores, removing any single method's ability to dominate the merge by scale alone.

**Automated retrieval evaluation.** `app/evaluate.py` runs a fixed set of test questions against the retrieval layer directly (not final LLM output, which is non-deterministic) and checks that relevant questions return sources above threshold, irrelevant questions return none, and sources are attributed to the correct source document. This caught the chunk-ID collision bug described above before it was noticed through manual testing.

**Dependency hygiene.** `requirements.txt` was found to contain a large number of packages unrelated to this project (Django, Flask, Kubernetes tooling, full CUDA/GPU PyTorch dependencies) after having been generated via `pip freeze` from a Python environment shared with other, unrelated work. This bloated Docker image builds to 1200+ seconds and pulled in over a gigabyte of unused NVIDIA/CUDA libraries for a project that runs LLM inference on CPU only. The fix was to rebuild the virtual environment from scratch, installing only the packages this project actually imports, and explicitly installing PyTorch's CPU-only build via its dedicated package index (`--extra-index-url https://download.pytorch.org/whl/cpu`) rather than letting pip resolve a default build that may include GPU support. The Dockerfile's `pip install` step was updated to reference the same index, since a `+cpu`-tagged package version exists only there, not on the default PyPI index.

**No hosted live demo.** Deploying to a platform like Streamlit Community Cloud would require restructuring the app into a single process (no Docker Compose, no separate FastAPI service), replacing local Ollama inference with a hosted LLM API, and accepting an ephemeral, non-persistent vector store. This was a deliberate trade-off against the project's core privacy guarantee rather than a limitation to fix — the local-only architecture is the point, not an obstacle. See "How to Use This App" near the top of this document for running it locally instead.

## Known Limitations

- Retrieval quality is still somewhat sensitive to query phrasing, though the advanced pipeline substantially mitigates this compared to single-query search.
- Retrieval parameters (`top_k`, similarity threshold, RRF's `k` constant, the vector/keyword weighting) were set from manual observation of this project's own test documents, not a formal evaluation set with ground-truth relevance labels.
- Generation latency is high on CPU-only hardware, and the advanced retrieval pipeline (multi-query generation plus hybrid search plus re-ranking) adds further latency on top of generation itself — an explicit, documented trade-off for retrieval accuracy. A cold-started local model is markedly slower than a subsequently "warm" one within the same session.
- `docker-compose.yml`'s `depends_on` guarantees container start order but not service readiness; a request sent to the backend immediately after `docker compose up` can fail before the API has finished initializing.
- The system has been tested with text-based PDFs; scanned or image-based documents would require an OCR step not yet implemented.

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Document ingestion, chunking, embeddings, vector storage | Complete |
| 2 | Retrieval, grounded prompting, local LLM generation | Complete |
| 3 | FastAPI backend with upload and query endpoints | Complete |
| 4 | Streamlit frontend | Complete |
| 5 | Containerization, similarity threshold, evaluation script | Complete |
| v2 – Advanced Retrieval | Multi-query generation, hybrid search, cross-encoder re-ranking | Complete |
| v2 – Evaluation & Grounding | Answer-level faithfulness checks | Planned |
| v2 – Observability | Structured logging, basic metrics | Planned |

## License

This project is intended as a portfolio and educational reference implementation.