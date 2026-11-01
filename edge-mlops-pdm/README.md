# Edge MLOps for Predictive Maintenance

Edge-to-cloud MLOps pipeline for industrial predictive maintenance.

## Architecture
- **Edge**: Lightweight model inference + data buffering + drift detection
- **Cloud**: Retraining pipeline, model registry (MLflow), promotion logic
- **Serving**: FastAPI model management + inference endpoints

## Key Features
- Continuous training with concept-drift awareness
- Model versioning & promotion (staging → production)
- Simulated edge device that pulls latest production model
- Asset health scoring for rotating / electrical assets

## Quick Start
```bash
pip install -r requirements.txt
python -m app.training.train          # initial train + MLflow
uvicorn app.api.main:app --reload     # model service
streamlit run app/dashboard/app.py
```
