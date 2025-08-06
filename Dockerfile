FROM nvidia/cuda:12.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.9 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for models
RUN mkdir -p /app/model_cache

# Pre-download TinyLlama model to embed it in the image
COPY download_model.py .
RUN python download_model.py

COPY . .

# Set model cache environment variable
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "app.py"]