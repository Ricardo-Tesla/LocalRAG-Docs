from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

reader = PdfReader("data/uploads/sample.pdf")

# Combine all pages into one string first
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(f"Total characters extracted: {len(full_text)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # roughly how many characters per chunk
    chunk_overlap=50,    # how many characters repeat between chunks
)

chunks = splitter.split_text(full_text)

print(f"Number of chunks: {len(chunks)}")
print("---First chunk---")
print(chunks[0])
print("---Second chunk---")
print(chunks[1])


model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode(chunks[0])

print(f"Embedding shape: {embedding.shape}")
print(f"First 10 numbers of the vector: {embedding[:10]}")