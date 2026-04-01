# ============================================================================
# LLMRAG Makefile
# ============================================================================

.PHONY: setup install ingest query serve test lint clean docker-build docker-up docker-down

# ── Setup ───────────────────────────────────────────────────────────────────
setup:
	@echo "🔧 Running LLMRAG setup..."
	python main.py setup
	@echo "✅ Setup complete"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install flake8 black isort mypy pytest-cov

# ── Pipeline Operations ────────────────────────────────────────────────────
ingest:
	python main.py ingest

query:
	@read -p "Question: " q; python main.py query "$$q"

serve:
	python main.py serve --port 8000

health:
	python main.py health

compliance:
	python main.py compliance

# ── Testing ─────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ --max-line-length=120 --ignore=E501,W503
	black --check src/ --line-length=120

format:
	black src/ --line-length=120
	isort src/

# ── Docker ──────────────────────────────────────────────────────────────────
docker-build:
	docker build -t llmrag:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f llmrag

# ── Local LLM (Ollama) ─────────────────────────────────────────────────────
ollama-up:
	docker-compose --profile local-llm up -d ollama
	@echo "Waiting for Ollama to start..."
	@sleep 5
	docker exec llmrag-ollama ollama pull llama3
	@echo "✅ Ollama ready with llama3"

# ── Cleanup ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	@echo "✅ Cleaned"

# ── Generate API Key ───────────────────────────────────────────────────────
gen-key:
	@read -p "Key name: " name; python main.py generate-key --name "$$name" --role admin
