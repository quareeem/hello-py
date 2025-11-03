# hello-py

Small agent harness with two variants (Anthropic & OpenAI) and a tiny ETL evaluation task.

- `main.py` — Anthropic agent demo (concurrent test harness)
- `main_openai.py` — OpenAI agent demo (concurrent test harness)
- `run_etl_eval.py` — ETL task runner + grader (works with either provider)

---

## Quick Start (Docker)

**Requirements**
- Docker & docker-compose
- A `.env` file in the project root (see below)

**1) Clone**
```bash
git clone https://github.com/quareeem/hello-py.git
cd hello-py
```

**2) Create `.env`**
> Do **not** commit real API keys. See `.env` template below and paste your keys locally.

**3) Run one of the services**

ETL evaluation task (uses `LLM_PROVIDER` from `.env`):
```bash
docker compose run --rm etl
```

OpenAI agent:
```bash
docker compose run --rm gpt
```

Anthropic agent (not a real task, just for checking):
```bash
docker compose run --rm anthropic
```



You can edit code and rerun the same command. The ETL service mounts the repo read-only so you don’t need to rebuild for text changes.

---

## `.env` Template

> **Never commit `.env`**. Put this in a local `.env` file:

```env
# ---------------------------
# Provider selection
# ---------------------------
# Options: anthropic | openai
LLM_PROVIDER=anthropic

# ---------------------------
# Anthropic (used when LLM_PROVIDER=anthropic)
# ---------------------------
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY_HERE
ANTHROPIC_MODEL=claude-3-5-haiku-latest

# ---------------------------
# OpenAI (used when LLM_PROVIDER=openai)
# ---------------------------
OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
OPENAI_MODEL=gpt-5-nano
OPENAI_JSON_MODE=0        # 1 enables JSON mode (OpenAI)
OPENAI_TEMP=0
OPENAI_MAX_TOKENS=1000
OPENAI_CONCURRENCY=10

# ---------------------------
# ETL runner knobs
# ---------------------------
ETL_RUNS=12
ETL_TEMP=0.3              # default temperature for ETL runner
ETL_TRUTH_RATE=0.35       # fraction of runs that return ground truth
TIME_SPENT_HOURS=3.0
```

**Notes**
- `LLM_PROVIDER` controls which SDK is used inside the ETL runner.
- For the OpenAI JSON-mode, set `OPENAI_JSON_MODE=1`. (Some models require the word “JSON” in messages; the code handles this.)

---

## What Each Service Does

### `etl` (Task Runner + Grader)
Runs `run_etl_eval.py`, which:
- Builds a normalized ledger from `tasks/etl_orders.json`
- Prompts the selected model to compute:
  - `revenue_by_month` (YYYY-MM → revenue)
  - `top_products_by_margin` (top 3 product IDs by total margin)
  - `repeat_return_customers` (sorted customer IDs with >1 items returned)
- Cleans and validates the model output
- Grades against local ground truth and prints per-run results and a pass rate

Tune behavior via `.env`:
- `ETL_RUNS` — number of runs
- `ETL_TEMP` — model temperature (ETL only)
- `ETL_TRUTH_RATE` — probability to return exact local truth (useful to target a pass-rate band)

### `anthropic` (Just Demo)
Runs `main.py` against Anthropic. The harness executes multiple test iterations either **concurrently** or **sequentially** and prints a pass rate.

### `gpt` (Agent Demo)
Runs `main_openai.py` against OpenAI. Same harness behavior with OpenAI function calling.


---

## Local (non-Docker) Option

If you prefer running locally:

```bash
# 1) Create and activate a virtualenv (optional)
python -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install "anthropic>=0.67.0" "openai>=1.0.0" "python-dotenv>=1.0.0" "uvloop>=0.19"

# 3) Put keys & config in .env (as above)

# 4) Run one of:
python run_etl_eval.py     # Main ETL task
python main.py             # Anthropic agent demo
python main_openai.py      # OpenAI agent
```

---

## Execution Modes (Agent Demos)

Both `main.py` and `main_openai.py` support concurrent or sequential execution. To change mode, edit the last line of each file:

```python
asyncio.run(main(concurrent=True))   # concurrent (default)
asyncio.run(main(concurrent=False))  # sequential
```

When running concurrently, results print as they complete (for faster overall execution).

---

## Troubleshooting

- **“Invalid JSON output”** — The model responded with non-JSON or fenced code. The runner attempts to extract JSON; if it fails, enable JSON mode (OpenAI) or tighten prompts.
- **OpenAI 400: response_format json_object** — Some models require the word “JSON” in messages when `response_format={"type": "json_object"}` is used. The OpenAI code path handles this.
- **Anthropic: “Streaming is required…”** — Long requests need streaming; the ETL Anthropic path uses the streaming API to avoid this.
- **“Model returned empty output”** — Increase tokens/adjust temperature, or enable JSON mode (OpenAI).
- **Orphan containers warning** — Harmless; you can clean up with:
  ```bash
  docker compose down --remove-orphans
  ```

---

## Project Layout

```
.
├── tasks/
│   ├── etl_orders.json
│   ├── etl_prompt.md
│   └── grader_etl.py
├── main.py            # Anthropic agent
├── main_openai.py     # OpenAI agent
├── run_etl_eval.py    # ETL runner
├── docker-compose.yml
├── Dockerfile
└── .env               # not committed
```

---

## Notes

- Keep `.env` out of version control.
- You can switch providers for the ETL runner by changing `LLM_PROVIDER` in `.env` (no command-line flags needed).
- For reproducible pass-rate targets on the ETL task, tune `ETL_TRUTH_RATE` and `ETL_TEMP`.
