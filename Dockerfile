# MeLi IAM Challenge - Dockerfile
FROM python:3.12-slim

# Metadata
LABEL maintainer="MeLi IAM Team"
LABEL description="MeLi IAM Assistant - Challenge MercadoLibre"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data/chroma /app/data/documents /app/data/processed

# Expose port
EXPOSE 8503

# Environment variables
ENV PYTHONPATH=/app
ENV LLM_MODEL=llama3:latest

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8503 || exit 1

# Start command
CMD ["python3", "start_meli_app.py"]