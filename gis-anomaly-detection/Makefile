.PHONY: install data train dashboard test

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	python -m app.data.generator

train:
	python -m app.pipeline.train

dashboard:
	streamlit run app/dashboard/app.py

test:
	pytest tests/ -v
