FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer until requirements change.
COPY requirements-app.txt .
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-app.txt

# Copy only what the service needs at runtime.
COPY src/__init__.py src/model.py src/
COPY app/ app/
COPY models/cnn_sqrt.pt models/cnn_sqrt.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'))"

# Hosts like Render pass the port in $PORT; fall back to 8000 for local use.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
