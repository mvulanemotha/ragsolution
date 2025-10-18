#!/bin/bash

# Activate virtualenv

source /services/ragsolution/venv/bin/activate


# Start FastAPI with Uvicorn
exec uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4