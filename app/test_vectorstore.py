from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- Step A: Load and chunk (same as before) ---
reader = PdfReader("data/uploads/sample.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)

# --- Step B: Embed ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks).tolist()  # convert to plain lists for Chroma

# --- Step C: Store in Chroma ---
client = chromadb.PersistentClient(path="chroma_db")  # saves to a folder on disk
collection = client.get_or_create_collection(name="documents")

# Chroma needs a unique ID for each chunk
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(
    ids=ids,
    embeddings=embeddings,
    documents=chunks,
)

print(f"Stored {collection.count()} chunks in the vector database.")

# --- Step D: Try a similarity search ---
query = "What are the key responsibilities?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2,
)

print("\n--- Top matching chunks for query ---")
for doc in results["documents"][0]:
    print(doc)
    print("---")