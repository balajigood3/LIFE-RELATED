#!/bin/bash
# Launch the Streamlit dashboard
cd "$(dirname "$0")/app"
streamlit run streamlit_app.py --server.port 8501 --server.headless true
