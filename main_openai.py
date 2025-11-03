# main_openai.py
import os, json, asyncio, re
from typing import Any, Callable, Dict, Tuple, List
from contextlib import redirect_stdout
from io import StringIO

from dotenv import load_dotenv


load_dotenv()  # reads .env

# ---------- Config ----------

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
NUM_RUNS = int(os.getenv("OPENAI_RUNS", "10"))
MAX_STEPS = int(os.getenv("OPENAI_MAX_STEPS", "5"))
TEMPERATURE = float(os.getenv("OPENAI_TEMP", "0"))
MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "64"))
CONCURRENCY = int(os.getenv("OPENAI_CONCURRENCY", "10"))
JSON_MODE = os.getenv("OPENAI_JSON_MODE", "0") == "1"   # <— add this
    # how many requests at once
# --- fast event loop (Linux container) ---
try:
    import uvloop  # type: ignore
    uvloop.install()
except Exception:
    pass

# --- one shared OpenAI client (reuse HTTP pool) ---
from openai import AsyncOpenAI


# Create ONE client and reuse it (persistent connections)
CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM = (
    "You are a helpful assistant. Use available tools when helpful. "
    "First, call the python_expression tool to compute the value, printing ONLY the number. "
    "Immediately afterward, you MUST call the submit_answer tool with that exact numeric result. "
    "Do not continue the conversation after submitting."
)


# ---------- Tool handlers ----------
def python_expression_tool(expression: str) -> Dict[str, Any]:
    """Exec arbitrary Python code (stdout captured)."""
    try:
        ns: Dict[str, Any] = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, ns, ns)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}

def submit_answer_tool(answer: Any) -> Dict[str, Any]:
    return {"answer": answer, "submitted": True}

TOOL_HANDLERS: Dict[str, Callable[..., Any]] = {
    "python_expression": python_expression_tool,
    "submit_answer": submit_answer_tool,
}

# OpenAI tool schemas (function calling)
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "python_expression",
            "description": "Evaluates a Python expression via exec(); stdout is returned. Use print(...) to emit output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to exec. Example: print((2**10 + 3**5) * 7 - 100)",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"description": "Final answer to submit"}},
                "required": ["answer"],
            },
        },
    },
]

# ---------- Agent loop ----------
async def run_agent_loop(
    prompt: str,
    max_steps: int = MAX_STEPS,
    verbose: bool = True,
) -> Any:
    """
    Mirrors the Anthropic main.py behavior but with OpenAI function-calling.
    """
    messages: List[Dict[str, Any]] = []
    if SYSTEM:
        sys_content = SYSTEM
        if JSON_MODE:
            # Satisfy JSON-mode requirement (if enabled)
            sys_content += " Respond with a JSON object. (JSON)"
        messages.append({"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": prompt})

    # Fallback if model computes but forgets to submit
    last_number: Any = None

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        # Step 1: force python_expression; later: let the model choose (so it can call submit_answer)
        tool_choice = (
            {"type": "function", "function": {"name": "python_expression"}}
            if step == 0 else "auto"
        )

        is_gpt5 = OPENAI_MODEL.startswith("gpt-5")
        kwargs = dict(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice=tool_choice,
        )
        if is_gpt5:
            kwargs["max_completion_tokens"] = MAX_TOKENS
        else:
            kwargs["max_tokens"] = MAX_TOKENS
            kwargs["temperature"] = TEMPERATURE

        if JSON_MODE:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = await CLIENT.chat.completions.create(**kwargs)
        except Exception as e:
            # If GPT-5 rejects temperature (or any param), drop it and retry once
            if "temperature" in str(e).lower():
                kwargs.pop("temperature", None)
                resp = await CLIENT.chat.completions.create(**kwargs)
            else:
                raise

        msg = resp.choices[0].message
        content = (msg.content or "").strip()
        tool_calls = msg.tool_calls or []

        if content and verbose:
            print(f"Assistant: {content}")

        submitted_answer = None

        if tool_calls:
            # Add assistant turn with tool calls
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                }
            )

            for tc in tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                if verbose:
                    print(f"\nUsing tool: {name}")
                    if name == "python_expression" and "expression" in args:
                        print("\nInput:\n```\n" + args["expression"] + "\n```")

                handler = TOOL_HANDLERS.get(name)
                result = {"error": f"Unknown tool: {name}"} if handler is None else (
                    handler(**args) if isinstance(args, dict) else handler(args)
                )

                # Send tool result back
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": name,
                        "content": json.dumps(result),
                    }
                )

                # Extract number from python_expression stdout for fallback
                if name == "python_expression":
                    stdout = ""
                    if isinstance(result, dict):
                        stdout = (result.get("result") or "").strip()
                    elif isinstance(result, str):
                        stdout = result.strip()

                    import re
                    m = re.findall(r"-?\d+(?:\.\d+)?", stdout)
                    if m:
                        txt = m[-1]
                        try:
                            last_number = int(txt) if re.fullmatch(r"-?\d+", txt) else float(txt)
                        except Exception:
                            last_number = txt  # keep as string if cast fails

                    # Nudge the model to submit now
                    if last_number is not None:
                        messages.append({"role": "user", "content": f"Use submit_answer with answer={last_number} now."})
                    else:
                        messages.append({"role": "user", "content": "Now call the submit_answer tool with the final number."})

                # Capture a submit
                if name == "submit_answer" and isinstance(result, dict) and "answer" in result:
                    submitted_answer = result["answer"]

            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer

            # If we just computed but model didn't submit, use fallback
            if last_number is not None:
                if verbose:
                    print(f"\nNo submit_answer call; returning parsed number: {last_number}")
                return last_number

        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return last_number  # final fallback (may be None)

# ---------- Test harness (same UX as Anthropic main.py) ----------
async def run_single_test(run_id: int, num_runs: int, prompt: str, expected_answer: Any, verbose: bool = False):
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")
    result = await run_agent_loop(prompt=prompt, verbose=verbose)
    success = result == expected_answer
    if success:
        print(f"✓ Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"✗ Run {run_id}: FAILURE - Got {result}, expected {expected_answer}")
    return run_id, success, result

async def main(concurrent: bool = True):
    num_runs = NUM_RUNS
    expected_answer = 8769
    prompt = "Calculate (2^10 + 3^5) * 7 - 100. Use the python_expression tool and then submit the answer."

    exec_mode = "concurrently" if concurrent else "sequentially"
    print(f"OpenAI model: {OPENAI_MODEL}")
    print(f"Running {num_runs} test iterations {exec_mode}…")
    print("=" * 60)

    tasks = [
        run_single_test(i + 1, num_runs, prompt, expected_answer, verbose=False)
        for i in range(num_runs)
    ]

    results = []
    if concurrent:
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
    else:
        for t in tasks:
            results.append(await t)

    successes = sum(1 for _, ok, _ in results if ok)
    pass_rate = successes / num_runs * 100.0

    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    asyncio.run(main(concurrent=True))
