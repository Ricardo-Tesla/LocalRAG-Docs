FROM python:3.12-slim

WORKDIR /app

# Copy requirements first, before the rest of the code. Docker caches each
# layer — if only application code changes (not dependencies), this layer
# is reused instead of reinstalling everything on every build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501