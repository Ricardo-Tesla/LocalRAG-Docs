from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# chunk_size=500 / overlap=50: keeps chunks small enough for precise
# retrieval while the overlap prevents meaning (e.g. a heading and the
# paragraph under it) from being split across chunk boundaries.
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def load_and_chunk_pdf(pdf_path: str):
    """
    Load a PDF and split it into chunks, tagging each with its source
    page and filename.

    Metadata is attached here, at ingestion time, rather than
    downstream — it's the only point where page and file identity are
    still available before chunks are merged into a shared vector store.

    Returns a list of dicts: {"text": ..., "page": ..., "source_file": ...}
    """
    reader = PdfReader(pdf_path)
    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text.strip():
            continue  # scanned/image-only pages produce no extractable text

        # Chunked per page, not on the full document text, so each
        # resulting chunk can be tagged with an accurate page number.
        page_chunks = splitter.split_text(page_text)

        for chunk_text in page_chunks:
            all_chunks.append({
                "text": chunk_text,
                "page": page_number,
                "source_file": Path(pdf_path).name,
            })

    return all_chunks