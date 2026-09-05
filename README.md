# LocalRAG Docs

A fully local, privacy-first Retrieval-Augmented Generation (RAG) system for technical document question answering. Users upload PDF documents, and the system retrieves relevant passages and generates grounded, source-attributed answers — entirely on-device, with no data sent to third-party APIs.

## Overview

LocalRAG Docs allows a user to upload one or more PDF documents and ask natural-language questions about their contents. The system retrieves the most relevant passages from the uploaded documents using semantic search, then passes those passages to a locally-hosted language model to generate an answer. Every answer is returned alongside the exact source passages, page numbers, and relevance scores used to produce it, allowing the user to verify the response against the original material.

The entire pipeline — document parsing, embedding generation, vector search, and language model inference — runs locally. No document content, queries, or generated answers leave the user's machine at any point.

## Motivation

Organizations in regulated or sensitive sectors (financial services, healthcare, industrial and manufacturing operations) frequently need to query internal technical documentation but cannot send proprietary or confidential content to third-party AI APIs. LocalRAG Docs demonstrates a production-oriented architecture for this exact requirement: a complete RAG pipeline with source transparency, built entirely on open-source, self-hostable components.

## Architecture

The system is composed of four layers, each with a single, well-defined responsibility:

| Layer | Component | Responsibility |
|---|---|---|
| Frontend | Streamlit | Document upload, question input, answer and source display |
| Backend API | FastAPI | Exposes ingestion and query logic over HTTP (`/upload`, `/query`, `/health`) |
| Retrieval & Generation | Custom RAG pipeline | Embeds queries, retrieves relevant chunks, constructs grounded prompts, calls the local LLM |
| Storage | ChromaDB | Persistent local vector store for document embeddings and metadata |

### Data flow

1. A user uploads a PDF through the Streamlit interface.
2. The file is sent to the FastAPI backend's `/upload` endpoint and saved to local disk.
3. The document is parsed page by page, split into overlapping text chunks, and each chunk is tagged with metadata (source filename, page number).
4. Each chunk is converted into a vector embedding and stored in a persistent ChromaDB collection.
5. When a user submits a question, the query is embedded using the same model, and ChromaDB returns the most semantically similar chunks (cosine similarity).
6. The retrieved chunks are inserted into a structured prompt instructing the language model to answer using only the supplied context.
7. The prompt is sent to a locally-running language model via Ollama, and the generated answer is returned to the frontend along with the source chunks, their page numbers, originating filenames, and similarity scores.

## Technology Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.12 |
| Document parsing | pypdf |
| Text chunking | LangChain (RecursiveCharacterTextSplitter) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
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
│   ├── rag.py             # Embedding, vector storage, retrieval, prompt construction, generation
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

## User Interface

The interface is deliberately structured around daily use rather than one-off queries. Document upload is placed in a persistent sidebar, since ingesting a document is an occasional action, while asking questions is the primary, repeated interaction and is given the full main viewport. This mirrors the layout conventions of persistent knowledge-base tools rather than single-purpose form-style demos.

The visual theme uses a warm, muted color palette (terracotta accents on a soft off-white background) configured through Streamlit's native theming system in `.streamlit/config.toml`, rather than ad hoc styling. This keeps every native widget — buttons, inputs, backgrounds — visually consistent without overriding Streamlit's component behavior.

## Setup and Installation

Two ways to run the application are supported: directly on the host machine, or fully containerized via Docker Compose. Docker Compose is the recommended path, since it starts all three services (API, frontend, and the LLM runtime) with a single command and requires no local Python environment setup.

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

The model persists in a named Docker volume, so this step is only needed once. The frontend will be available at `http://localhost:8501`, and the API at `http://localhost:8000`.

### Option B: Run directly on the host

**Prerequisites:**
- Python 3.10 or later
- [Ollama](https://ollama.com) installed locally

```bash
git clone <repository-url>
cd localrag-docs

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
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

The API will be available at `http://127.0.0.1:8000` (interactive documentation at `/docs`), and the user interface at `http://localhost:8501`.

### Running the evaluation script

A small set of regression checks validates that retrieval behaves correctly against known documents — that relevant questions return sources above the similarity threshold, that irrelevant questions correctly return none, and that sources are attributed to the correct document:

```bash
python -m app.evaluate
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns service status |
| `/upload` | POST | Accepts a PDF file, ingests it into the vector store |
| `/query` | POST | Accepts a JSON body `{"question": "..."}`, returns a generated answer and its supporting sources |

Full request and response schemas are available via the auto-generated Swagger UI at `/docs`.

## Design Notes

**Cosine similarity.** The vector store is explicitly configured to use cosine distance rather than Chroma's default (Euclidean/L2), which is the appropriate metric for comparing the semantic similarity of text embeddings.

**Per-chunk metadata.** Page number and source filename are attached to each chunk at ingestion time, before documents are combined into a shared vector index. This allows multiple documents to coexist in a single collection while preserving the ability to trace every answer back to its exact origin, and prevents retrieval from one document being diluted by irrelevant content from another.

**Grounded prompting.** The system prompt explicitly instructs the model to answer only from the retrieved context and to state when the context is insufficient, reducing (though not eliminating) the risk of hallucinated answers.

**Similarity threshold.** Chroma always returns the closest available vectors regardless of how weak the actual match is, which previously meant a clearly unrelated question could still surface several low-confidence source chunks alongside a correct "I don't know" answer. A minimum cosine similarity threshold (0.2, set empirically from observed scores across relevant and irrelevant queries) filters these out before they reach either the language model or the user, and the pipeline returns an explicit "not enough information" response when no chunk clears it — without calling the LLM at all in that case.

**Globally unique chunk identifiers.** Chunk IDs were originally generated per-ingestion (`chunk_0`, `chunk_1`, ...), which caused chunks from a newly ingested document to silently overwrite same-indexed chunks from a previously ingested document in the vector store, since Chroma treats a duplicate ID as an update. This was caught by the evaluation script, not manual testing. IDs are now prefixed with a UUID per ingestion call, guaranteeing uniqueness across all documents.

**Automated retrieval evaluation.** `app/evaluate.py` runs a fixed set of test questions against the retrieval layer (not the LLM output, which is non-deterministic and harder to assert on) and checks that relevant questions return sources above threshold, irrelevant questions return none, and sources are attributed to the correct source document. This is a regression check intended to be re-run after any change to chunking, embeddings, or thresholds.

## Known Limitations

- Retrieval quality is sensitive to query phrasing. Questions containing language about the interaction itself (e.g. "according to the document uploaded") introduce noise into the embedding and can reduce match quality compared to direct, content-focused questions.
- The current retrieval count (`top_k = 5`) and similarity threshold (`0.2`) were set from manual observation of this project's own test documents, not a formal evaluation set with ground-truth relevance labels.
- Generation latency is high on CPU-only hardware (the target environment for this project, in keeping with its fully local design). A local language model running without GPU acceleration, further slowed by container virtualization, can take on the order of minutes to respond. Scaling this to production use would require GPU-accelerated inference, a smaller/quantized model, or a dedicated inference server.
- `docker-compose.yml`'s `depends_on` guarantees container start order but not service readiness; a request sent to the backend immediately after `docker compose up` can fail before the API has finished initializing. A production setup would add explicit health checks.
- The system has been tested with text-based PDFs; scanned or image-based documents would require an OCR step not yet implemented.

## Status

| Phase | Description | Status |
|---|---|---|
| 1 | Document ingestion, chunking, embeddings, vector storage | Complete |
| 2 | Retrieval, grounded prompting, local LLM generation | Complete |
| 3 | FastAPI backend with upload and query endpoints | Complete |
| 4 | Streamlit frontend | Complete |
| 5 | Containerization, similarity threshold, evaluation script | Complete |
| 6 | Multi-collection support, logging, testing | Planned |

## License

This project is intended as a portfolio and educational reference implementation.