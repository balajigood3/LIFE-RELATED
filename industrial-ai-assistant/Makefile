.PHONY: install run demo test lint

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	streamlit run app/main.py

demo:
	python -m app.demo

test:
	pytest tests/ -v

lint:
	ruff check app/ tests/
