from pathlib import Path
import os
import json
from typing import Tuple
from tasks.grader_etl import grade

TASK_PROMPT_PATH = Path("tasks/etl_prompt.md")
DATA_PATH = Path("tasks/etl_orders.json")

SYSTEM = (
  "You are a meticulous data analyst. "
  "Follow the instructions exactly and return ONLY valid JSON. "
  "Interpret 'margin' as TOTAL gross margin across all units (sales minus returns), not per-unit."
)


def build_user_prompt() -> str:
    prompt_text = TASK_PROMPT_PATH.read_text()
    data = json.loads(DATA_PATH.read_text())
    return f"{prompt_text}\n\n### Orders JSON\n```json\n{json.dumps(data, indent=2)}\n```"

def pick_provider() -> str:
    pref = os.getenv("LLM_PROVIDER", "").lower()
    if pref in ("anthropic", "openai"):
        return pref
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError("No API keys found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")

def call_model(prompt: str, temperature: float, max_tokens: int) -> str:
    provider = pick_provider()
    try:
        if provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            model = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet")
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            try:
                parts = getattr(msg, "content", msg)
                text = "".join([b.get("text", "") for b in parts if isinstance(b, dict)])
            except Exception:
                text = str(msg)
        else:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt}
                ]
            )
            try:
                text = resp.choices[0].message.content
            except Exception:
                try:
                    text = resp["choices"][0]["message"]["content"]
                except Exception:
                    text = str(resp)
    except Exception as e:
        raise RuntimeError(f"Model call failed: {e}")

    if not text or not str(text).strip():
        raw = repr(locals().get("msg", locals().get("resp", "")))
        raise RuntimeError(f"Model returned empty output. Raw response: {raw[:1000]!r}")

    return str(text)

def run_once(temperature: float = 1.0, max_tokens: int = 10_000) -> Tuple[bool, str]:
    try:
        text = call_model(build_user_prompt(), temperature=temperature, max_tokens=max_tokens)
        print("DEBUG: model output repr:", repr(text)[:1000])
        result = grade(text, str(DATA_PATH))
        return result["passed"], result["reason"]
    except Exception as e:
        return False, f"Model call error: {e}"

def main(n_runs: int = 12, temperature: float = 1.0):
    passed = 0
    start = time.time()
    
    for i in range(n_runs):
        success, reason = run_once(temperature=temperature)
        passed += int(success)
        print(f"[Run {i+1:02d}] {'PASS' if success else 'FAIL'} — {reason}")
    
    elapsed = time.time() - start
    rate = passed / n_runs
    print(f"\nPass rate: {passed}/{n_runs} = {rate:.1%}  |  Elapsed: {elapsed:.1f}s")
    print("Self-reported time spent building this task: 3.0 hours")

if __name__ == "__main__":
    import time
    main()
