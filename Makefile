.PHONY: help install test train generate web clean lint format

help:
	@echo "Available commands:"
	@echo "  make install    - Install package and dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make train      - Train a model with default settings"
	@echo "  make generate   - Generate text from trained model"
	@echo "  make web        - Start web interface"
	@echo "  make clean      - Clean generated files"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code with black"

install:
	pip install -e .

test:
	python test_microgpt.py

train:
	python cli.py train --config config.yaml --experiment-name default

generate:
	python cli.py generate --temperature 0.7 --num-samples 10

web:
	python web_app.py

clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf checkpoints/*.pkl checkpoints/*.json
	rm -rf logs/*.jsonl
	rm -rf build dist *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

lint:
	flake8 *.py --max-line-length=100 --ignore=E203,W503

format:
	black *.py --line-length=100
