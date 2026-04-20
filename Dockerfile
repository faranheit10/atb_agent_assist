# Multi-stage Dockerfile for ATB Agent Assist
# -------------------------------------------

# Stage 1: Build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies (for packages like chromadb if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app code
COPY . .

# Ensure data and chroma_db directories exist for persistence
RUN mkdir -p data/kb chroma_db

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PORT=8001

# Expose the API/Frontend port
EXPOSE 8001

# Start the server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
