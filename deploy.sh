#!/usr/bin/env bash
# One-command deploy on a Linux server with Docker. Run from the repo folder: ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"

command -v docker >/dev/null || { echo "Docker is not installed: https://docs.docker.com/engine/install/"; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "Docker Compose plugin missing (apt install docker-compose-plugin)"; exit 1; }

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo "Created .env with the free setup (local embeddings + Ollama)."
  echo "Review it (nano .env), then run ./deploy.sh again."
  exit 0
fi

env_get() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }
need() { [ -n "$(env_get "$1")" ] || { echo "$1 is empty in .env ($2)"; exit 1; }; }

[ "$(env_get EMBEDDING_PROVIDER)" = "openai" ] && need OPENAI_API_KEY "needed for OpenAI embeddings"
[ "$(env_get LLM_PROVIDER)" = "anthropic" ] && need ANTHROPIC_API_KEY "needed for Claude"
[ "$(env_get RERANKER)" = "cohere" ] && need COHERE_API_KEY "needed for Cohere rerank"

profile=()
base_url=$(env_get LLM_BASE_URL)
if [[ "$base_url" == *ollama* ]]; then
  profile=(--profile ollama)
elif [ "$(env_get LLM_PROVIDER)" = "openai" ] && [ -n "$base_url" ]; then
  need LLM_API_KEY "needed for $base_url"
fi

docker compose "${profile[@]}" up -d --build

if [ ${#profile[@]} -gt 0 ]; then
  model=$(env_get LLM_MODEL)
  echo "Pulling Ollama model $model (first time only, a few GB)..."
  docker compose exec -T ollama ollama pull "$model"
fi

echo "Waiting for the API..."
for _ in $(seq 1 60); do
  if docker compose exec -T app python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; then
    break
  fi
  sleep 2
done

if ls data/corpus/*.pdf >/dev/null 2>&1; then
  echo "Ingesting (first run downloads the embedding model)..."
  docker compose exec -T app python -m rag ingest data/corpus
else
  echo "No PDFs in data/corpus yet. Copy your manuals there, then run: docker compose exec app python -m rag ingest data/corpus"
fi

docker compose ps
bind=$(env_get APP_BIND); port=$(env_get APP_PORT)
echo "Up. UI: http://${bind:-127.0.0.1}:${port:-8000}"
