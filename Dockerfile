FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
COPY prompts ./prompts
COPY eval ./eval
ENV DATA_DIR=/app/data PROMPT_PATH=/app/prompts/assistant.yaml GOLDEN_PATH=/app/eval/golden.jsonl

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "retail_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
