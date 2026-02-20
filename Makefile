.PHONY: help install install-dev test test-cov lint format clean docker-build docker-run docs

help:
	@echo "microgpt Makefile commands:"
	@echo "  install      - Install package"
	@echo "  install-dev  - Install with dev dependencies"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  integration  - Run integration tests"
	@echo "  lint         - Run linters"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  train        - Quick training run"
	@echo "  generate     - Quick generation"
	@echo "  chat         - Start chat interface"
	@echo "  server       - Start API server"
	@echo "  profile      - Profile model performance"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,all]"

install-all:
	pip install -e ".[all]"

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=microgpt --cov-report=html --cov-report=term

integration:
	python integration_test.py

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

format:
	black . --line-length=100

type-check:
	mypy . --ignore-missing-imports

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

docker-build:
	docker build -t microgpt:latest .

docker-run:
	docker run -it --rm -p 5000:5000 microgpt:latest

train:
	python main.py train --epochs 100

generate:
	python main.py generate --num-samples 5

chat:
	python main.py chat

server:
	python api_server.py

profile:
	python -c "from profiling import profile_model; from model import GPT; m = GPT(256, 16, 2, 32, 4); profile_model(m, 'all')"

benchmark:
	python benchmark.py

zoo:
	python main.py zoo

check: lint test integration
	@echo "All checks passed!"

all: install-dev check
	@echo "Setup complete!"

.DEFAULT_GOAL := help
