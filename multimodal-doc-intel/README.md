# Multimodal Industrial Document Intelligence

Vision + Text RAG system for industrial technical documents, single-line diagrams, nameplates, and vendor manuals.

## Capabilities
- Ingest PDFs and images of manuals / SLDs / nameplates
- OCR + Vision-Language understanding
- Graph-augmented / vector RAG over technical content
- Query interface for setpoints, LOTO points, protection settings, etc.

## Quick Start
```bash
pip install -r requirements.txt
# Place sample PDFs/images in data/sample_docs and data/sample_images
streamlit run app/dashboard/app.py
# or
uvicorn app.api.main:app --reload
```

Note: For full vision capabilities set OPENAI_API_KEY (GPT-4o) or configure a local VLM.
