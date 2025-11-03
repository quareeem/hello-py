Analyze the provided orders data and return a JSON object with exactly these keys:

1) "revenue_by_month": Monthly revenue in YYYY-MM format.
   - For each order: order_total = sum(unit_price * quantity) over all items.
   - For each return on that SAME order: return_total += (unit_price * quantity) using the item’s unit_price from that order.
   - For the month equal to the order’s date (not the return date), add (order_total - return_total) to that month.
   - Round to 2 decimals per month at the end.

2) "top_products_by_margin": Top 3 product_ids by TOTAL gross margin, sorted by margin descending.
   - For each item: margin += (unit_price - unit_cost) * quantity
   - For each return: margin -= (unit_price - unit_cost) * quantity (use the item’s unit_price and unit_cost from that order)
   - If margins tie, sort those product_ids ascending.

3) "repeat_return_customers": Unique customer_ids who had ANY returns (quantity > 0) in any of their orders, sorted ascending.

Output ONLY a JSON object with exactly these three keys. No markdown, no code fences, no prose.
