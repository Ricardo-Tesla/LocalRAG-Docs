from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from app.rag import generate_answer, ingest_pdf
from pathlib import Path
import shutil

app = FastAPI(title="LocalRAG Docs API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    text: str
    page: int
    source_file: str
    similarity_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = generate_answer(request.question)
    return result

UPLOAD_DIR = Path("data/uploads")


@app.post("/upload")
def upload_document(file: UploadFile):
    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    ingest_pdf(str(save_path))

    return {"filename": file.filename, "status": "ingested"}