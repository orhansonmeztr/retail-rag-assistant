#!/usr/bin/env bash
# Copy the shared package next to function_app.py so `func azure functionapp publish`
# uploads it with the app (Functions builds only what is inside this folder).
set -euo pipefail
cd "$(dirname "$0")"
rm -rf retail_rag
cp -r ../src/retail_rag ./retail_rag
find retail_rag -name __pycache__ -type d -prune -exec rm -rf {} +
echo "ready: run 'func azure functionapp publish <app-name> --python' from $(pwd)"
