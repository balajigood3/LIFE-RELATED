#!/usr/bin/env bash
# Simple launcher for the Multi-Agent Research System

set -e

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check for .env
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env and add your OPENAI_API_KEY and SERPER_API_KEY"
    exit 1
fi

# Run
python main.py "$@"
