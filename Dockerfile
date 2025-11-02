FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "anthropic>=0.67.0" "openai>=1.0.0" "python-dotenv>=1.0.0"
COPY tasks ./tasks
COPY run_etl_eval.py .
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "run_etl_eval.py"]
