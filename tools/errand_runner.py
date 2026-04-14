# ============================================================
# tools/errand_runner.py
# Generates an errand runner quote: calculates the service fee
# based on number of stores, estimated time, and items count.
#
# Future: integrate with a real marketplace API (TaskRabbit, etc.)
# ============================================================

from config.settings import ERRAND_BASE_FEE, ERRAND_PER_STORE_FEE


def generate_errand_quote(
    shopping_plan: dict,
    route_plan: dict,
    tip_pct: float = 0.15,
) -> dict:
    """
    Given the optimized shopping plan and route, generate a cost
    estimate for an errand runner to do the shopping on your behalf.

    Args:
        shopping_plan: output from price_optimizer.optimize_shopping_list()
        route_plan:    output from route_planner.plan_route()
        tip_pct:       suggested tip as a fraction of groceries total (default 15%)

    Returns a quote dict with line-item breakdown.
    """
    num_stores = len(shopping_plan["store_ids"])
    groceries_total = shopping_plan["total_cost"]

    # Service fee: base + per-store surcharge
    service_fee = ERRAND_BASE_FEE + (num_stores * ERRAND_PER_STORE_FEE)

    # Suggested tip based on grocery subtotal
    suggested_tip = round(groceries_total * tip_pct, 2)

    grand_total = round(groceries_total + service_fee + suggested_tip, 2)

    # Estimate runner time: route duration + 10 min per store for shopping
    shopping_time_min = num_stores * 10
    drive_time_min = route_plan.get("total_duration_min") or 0
    estimated_total_min = int(shopping_time_min + drive_time_min)

    return {
        "groceries_subtotal": groceries_total,
        "service_fee": round(service_fee, 2),
        "suggested_tip": suggested_tip,
        "grand_total": grand_total,
        "estimated_time_min": estimated_total_min,
        "num_stores": num_stores,
        "breakdown": {
            "base_fee": ERRAND_BASE_FEE,
            "per_store_fee": ERRAND_PER_STORE_FEE,
            "stores_visited": num_stores,
        },
        "note": "Tip is a suggestion. Final cost may vary if item prices changed.",
    }
