# Task: Tiny ETL — Nested JSON → Answers

You are given a small nested JSON of orders. **Compute exactly these three outputs** and return **ONLY a JSON object**, no prose:

1. **`revenue_by_month`**: object mapping `"YYYY-MM"` → total revenue **(sum of unit_price × quantity)** for that month, **excluding returned quantities**. Round to 2 decimals.

2. **`top_products_by_margin`**: array of the **top 3 product_ids by total gross margin**, where  
   margin = Σ[(unit_price − unit_cost) × (sold_quantity − returned_quantity)] per product across all orders.  
   Sort **descending by total margin**. **Tie-breaker**: (a) higher total revenue for the tied products; (b) if still tied, lexicographic by product_id.

3. **`repeat_return_customers`**: array of `customer_id` values for customers who **returned more than 1 item in total across all orders**. Sort ascending and de-duplicate.

### Input rules
- Use the JSON provided below as the only data source.
- The `"returns"` array lists `{product_id, quantity}` returned from that order. Returned items must be **subtracted** from both revenue and margin.
- Assume each returned `product_id` appears in that order’s `items` with matching `unit_price` and `unit_cost`.
- Output must be **valid JSON** with **exactly** the keys: `revenue_by_month`, `top_products_by_margin`, `repeat_return_customers`.

### Output schema
```json
{
  "revenue_by_month": {"YYYY-MM": 0.0, "...": 0.0},
  "top_products_by_margin": ["P-...","P-...","P-..."],
  "repeat_return_customers": ["C-...", "..."]
}
