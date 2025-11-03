# grader_etl.py
from __future__ import annotations

import json
from typing import Dict, List, Tuple, Any

# Float comparison tolerance
TOL = 1e-2  # 0.01 tolerance for floats


def _month(s: str) -> str:
    """
    Extract YYYY-MM from a date string that is expected to be YYYY-MM-DD.
    """
    return s[:7]


def _compute_truth(data: Dict[str, Any]) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Compute the ground-truth answers from raw order data.

    Returns:
        revenue_by_month: mapping YYYY-MM -> revenue rounded later
        top3: top 3 product_ids by total margin (desc), then revenue (desc), then id (asc)
        repeat: customers with total returned quantity > 1 (sorted ascending)
    """
    orders = data["orders"]

    revenue_by_month: Dict[str, float] = {}
    margin_by_product: Dict[str, float] = {}
    revenue_by_product: Dict[str, float] = {}
    returns_by_customer: Dict[str, int] = {}

    for o in orders:
        m = _month(o["order_date"])
        # Ensure month bucket exists
        revenue_by_month.setdefault(m, 0.0)

        # Accumulate sales (revenue and margin)
        for it in o["items"]:
            pid = it["product_id"]
            price = float(it["unit_price"])
            cost = float(it["unit_cost"])
            qty = int(it["quantity"])

            revenue_by_month[m] += price * qty
            margin_by_product[pid] = margin_by_product.get(pid, 0.0) + (price - cost) * qty
            revenue_by_product[pid] = revenue_by_product.get(pid, 0.0) + price * qty

        # Accumulate returns (subtract revenue and margin) and tally per customer
        for r in o.get("returns", []):
            pid = r["product_id"]
            rqty = int(r["quantity"])

            # Match the exact item's unit_price/unit_cost from the same order
            match = next(it for it in o["items"] if it["product_id"] == pid)
            price = float(match["unit_price"])
            cost = float(match["unit_cost"])

            revenue_by_month[m] -= price * rqty
            margin_by_product[pid] -= (price - cost) * rqty
            revenue_by_product[pid] -= price * rqty

            cid = o["customer_id"]
            returns_by_customer[cid] = returns_by_customer.get(cid, 0) + rqty

    # Sort key: margin desc, revenue desc, id asc
    def sort_key(pid: str):
        return (-margin_by_product[pid], -revenue_by_product.get(pid, 0.0), pid)

    top3 = sorted(margin_by_product.keys(), key=sort_key)[:3]

    # Repeat return customers (>1 total returned items), sorted asc
    repeat = sorted([cid for cid, total in returns_by_customer.items() if total > 1])

    # Round revenue_by_month to 2 decimals for comparison
    revenue_by_month = {k: round(v + 1e-12, 2) for k, v in revenue_by_month.items()}

    return revenue_by_month, top3, repeat


def _extract_json(s: str):
    """
    Accept pure JSON or best-effort extraction of the first {...} block.
    Mirrors the original permissive behavior.
    """
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        # Heuristic extraction: take first '{' .. last '}'
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
        # Preserve original raise behavior if no valid JSON segment
        raise


def grade(model_output_text: str, data_json_path: str = "tasks/etl_orders.json") -> Dict[str, Any]:
    """
    Grade the model output text against truth derived from data_json_path.

    Returns:
        {
          "passed": bool,
          "reason": str
        }
    """
    with open(data_json_path, "r") as f:
        data = json.load(f)

    truth_rev, truth_top3, truth_repeat = _compute_truth(data)

    # Parse model output
    try:
        pred = _extract_json(model_output_text)
    except Exception as e:
        return {"passed": False, "reason": f"Invalid JSON output: {e}"}

    # Schema checks
    if not isinstance(pred, dict):
        return {"passed": False, "reason": "Top-level output must be a JSON object."}

    required_keys = {"revenue_by_month", "top_products_by_margin", "repeat_return_customers"}
    if set(pred.keys()) != required_keys:
        return {"passed": False, "reason": f"Output keys must be exactly {sorted(required_keys)}."}

    # 1) revenue_by_month
    rev = pred["revenue_by_month"]
    if not (isinstance(rev, dict) and all(isinstance(k, str) for k in rev.keys())):
        return {
            "passed": False,
            "reason": "revenue_by_month must be an object mapping YYYY-MM to number.",
        }

    # All expected months present & values within tolerance
    for k, v_true in truth_rev.items():
        v_pred = rev.get(k, None)
        if v_pred is None or not isinstance(v_pred, (int, float)):
            return {"passed": False, "reason": f"revenue_by_month missing or invalid for {k}."}
        if abs(float(v_pred) - v_true) > TOL:
            return {"passed": False, "reason": f"Revenue mismatch for {k}: got {v_pred}, expected {v_true}."}

    # No extra months that aren't in truth
    if set(rev.keys()) != set(truth_rev.keys()):
        return {"passed": False, "reason": "revenue_by_month has unexpected month keys."}

    # 2) top_products_by_margin
    top = pred["top_products_by_margin"]
    if not (isinstance(top, list) and len(top) == 3 and all(isinstance(x, str) for x in top)):
        return {
            "passed": False,
            "reason": "top_products_by_margin must be an array of 3 product_id strings.",
        }
    if top != truth_top3:
        return {
            "passed": False,
            "reason": f"Incorrect top_products_by_margin: got {top}, expected {truth_top3}.",
        }

    # 3) repeat_return_customers
    rpt = pred["repeat_return_customers"]
    if not (isinstance(rpt, list) and all(isinstance(x, str) for x in rpt)):
        return {
            "passed": False,
            "reason": "repeat_return_customers must be an array of customer_id strings.",
        }
    if rpt != truth_repeat:
        return {
            "passed": False,
            "reason": f"Incorrect repeat_return_customers: got {rpt}, expected {truth_repeat}.",
        }

    return {"passed": True, "reason": "All checks passed."}
