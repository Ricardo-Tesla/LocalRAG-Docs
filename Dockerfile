FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK stopwords at build time, not runtime — keeps the running
# container free of network dependencies, consistent with the project's
# fully-local design (the container's Ollama/BM25/embedding pipeline
# shouldn't need internet access to function once built).
RUN python -m nltk.downloader stopwords

COPY . .

EXPOSE 8000 8501