FROM python:3.13-slim
WORKDIR /app

# minimal system deps + a tiny init
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tini && rm -rf /var/lib/apt/lists/*

# SDKs + .env loader
RUN pip install --no-cache-dir "anthropic>=0.67.0" "openai>=1.0.0" "python-dotenv>=1.0.0" "uvloop>=0.19"

# project files
COPY tasks ./tasks
COPY main.py .
COPY main_openai.py .
# keep this if you still want the ETL runner available:
COPY run_etl_eval.py .

ENTRYPOINT ["/usr/bin/tini", "--"]
# default = Anthropic path (their main.py)
CMD ["python", "main.py"]
