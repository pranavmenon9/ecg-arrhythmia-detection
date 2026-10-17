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
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
