.PHONY: install samples app api test lint format evaluate eval-retrieval docker

install:            ## Create the environment with all extras
	python -m pip install -e ".[ui,dev]"

samples:            ## Index the 32 sample documents
	knowledge-assistant ingest --samples

app:                ## Start the web interface
	streamlit run app/streamlit_app.py

api:                ## Start the HTTP API
	uvicorn knowledge_assistant.api:app --reload

test:               ## Run the automated tests (offline)
	pytest

lint:               ## Check code style
	ruff check . && ruff format --check .

format:
	ruff format . && ruff check --fix .

eval-retrieval:     ## Retrieval metrics only (embeddings, no answer generation)
	knowledge-assistant evaluate --retrieval-only

evaluate:           ## Full evaluation (needs GEMINI_API_KEY; free tier: keep the delay)
	knowledge-assistant evaluate --delay 4

docker:
	docker build -t knowledge-assistant .
