from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def load_and_chunk_pdf(pdf_path: str):
    """
    Load a PDF and split it into chunks, tracking which page
    each chunk came from.

    Returns a list of dicts: {"text": ..., "page": ...}
    """
    reader = PdfReader(pdf_path)
    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if not page_text.strip():
            continue  # skip blank pages

        page_chunks = splitter.split_text(page_text)

        for chunk_text in page_chunks:
            all_chunks.append({
                "text": chunk_text,
                "page": page_number,
            })

    return all_chunks