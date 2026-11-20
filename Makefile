.PHONY: up down ingest eval-retrieval eval-generation demo test lint

up:            ## start Postgres + API on http://localhost:8000
	docker compose up -d --build

down:
	docker compose down

ingest:        ## ingest every PDF in data/corpus
	docker compose exec app python -m rag ingest data/corpus

eval-retrieval:
	docker compose exec app python -m eval.run_eval retrieval

eval-generation:
	docker compose exec app python -m eval.run_eval generation --limit 30

demo:          ## no API keys needed: sample manual, fake embeddings, fake LLM
	docker compose exec -e EMBEDDING_PROVIDER=fake -e EMBEDDING_DIM=256 -e LLM_PROVIDER=fake app \
		sh -c "python -m rag reset && python -m rag ingest data/sample && \
		python -m eval.run_eval retrieval --questions eval/questions.sample.jsonl"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .
