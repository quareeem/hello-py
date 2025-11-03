FROM python:3.13-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tini && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    "anthropic>=0.67.0" \
    "openai>=1.0.0" \
    "python-dotenv>=1.0.0" \
    "uvloop>=0.19"

COPY tasks ./tasks
COPY main.py .
COPY main_openai.py .
COPY run_etl_eval.py .

ENTRYPOINT ["/usr/bin/tini", "--"]
# Default = Anthropic path (their main.py)
CMD ["python", "main.py"]
