FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# "local" pulls sentence-transformers for free CPU embeddings and reranking.
# Build with --build-arg EXTRAS=cohere for a smaller image that only uses APIs.
ARG EXTRAS="cohere,local"

# 1) Dependencies only. This heavy layer (torch etc.) is cached and reused until
#    pyproject.toml changes, so code edits do not reinstall ~1 GB of libraries.
COPY pyproject.toml ./
RUN mkdir -p rag eval && touch rag/__init__.py eval/__init__.py \
 && if echo "$EXTRAS" | grep -q local; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu; fi \
 && pip install ".[${EXTRAS}]" \
 && pip uninstall -y rag-manuals \
 && rm -rf rag eval

# 2) The code: a small layer that changes often
COPY rag ./rag
COPY eval ./eval
COPY scripts ./scripts
RUN pip install --no-deps --no-build-isolation . 2>/dev/null || pip install --no-deps .

RUN useradd --create-home app && chown -R app /app && mkdir -p /home/app/.cache && chown app /home/app/.cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
