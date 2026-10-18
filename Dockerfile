# Nexora Knowledge Assistant - Streamlit application
# Build:  docker build -t knowledge-assistant .
# Run:    docker run -p 8501:8501 -e GEMINI_API_KEY=your-key -v ka-data:/app/data/index knowledge-assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so that code changes do not invalidate this layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install -e ".[ui]"

COPY app ./app
COPY .streamlit ./.streamlit
COPY data/sample_documents ./data/sample_documents

# Run as a non-root user; the index and the uploaded files live in /app/data.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data/index /app/data/storage \
    && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
