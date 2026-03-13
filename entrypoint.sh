#!/bin/bash
set -e

APP_MODE="${APP_MODE:-jupyter}"

if [ "$APP_MODE" = "gradio" ]; then
    echo "Starting Gradio demo on port 7860..."
    python app/grapholab_demo.py
elif [ "$APP_MODE" = "jupyter" ]; then
    echo "Starting JupyterLab on port 8888..."
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.token="${JUPYTER_TOKEN}" \
        --notebook-dir=/app/notebooks
else
    echo "Unknown APP_MODE: $APP_MODE. Use 'jupyter' or 'gradio'."
    exit 1
fi
