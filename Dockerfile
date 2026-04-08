FROM python:3.11-slim

# Create non-root user required by Hugging Face Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install deps before copying source (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and fix ownership in one layer
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

CMD ["python", "server.py"]
