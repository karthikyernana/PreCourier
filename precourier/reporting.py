"""Stable merchant reports built from persisted outcomes and ledger entries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ledger import margin_on_delivery
from .money import PPM, format_paise
from .store.repo import ActionRepo, DecisionRepo, LedgerRepo, OrderRepo, OutcomeRepo
from .tenancy import Tenant


REPORT_VERSION = "roi-v1"


def roi_report(tenant: Tenant, *, arm: str = "live",
               since: Optional[str] = None) -> Dict[str, Any]:
    """Return a reconciled, tenant-scoped ROI report.

    The report intentionally distinguishes realised ledger balance from a
    counterfactual recovery estimate. Without a control arm, calling every positive
    balance "recovered" would be a marketing claim rather than an accounting fact.
    """
    ledger_repo = LedgerRepo(tenant)
    outcomes = OutcomeRepo(tenant).counts(since=since)
    decisions = DecisionRepo(tenant).counts_by_action(since=since)
    by_reason = ledger_repo.by_reason(arm=arm)
    net = ledger_repo.net(arm=arm)
    evaluated = sum(decisions.values())
    action_spend = -sum(
        amount for reason, amount in by_reason.items()
        if reason in ("action_cost", "incentive") and amount < 0)
    return {
        "report_version": REPORT_VERSION,
        "provenance": "persisted_merchant_outcomes",
        "synthetic_data": False,
        "merchant_id": tenant.merchant_id,
        "arm": arm,
        "since": since,
        "orders_evaluated": evaluated,
        "outcomes": outcomes,
        "decisions_by_action": decisions,
        "ledger_by_reason_paise": by_reason,
        "ledger_by_reason_rupees": {
            reason: format_paise(amount, sign=True)
            for reason, amount in by_reason.items()},
        "net_paise": net,
        "net_rupees": format_paise(net, sign=True),
        "action_spend_paise": action_spend,
        "action_spend_rupees": format_paise(action_spend),
        "returns_prevented": None,
        "cost_per_return_prevented_paise": None,
        "assumptions": [
            "Realised balance is not incremental recovery without a control arm.",
            "Synthetic and projected results are reported separately from merchant outcomes.",
        ],
        "exceptions": _exceptions(tenant, since=since),
    }


def _exceptions(tenant: Tenant, *, since: Optional[str]) -> List[Dict[str, Any]]:
    """Return a small, privacy-conscious list of inspectable decision exceptions."""
    sql = (
        "SELECT d.order_id, d.action, d.reason, d.expected_delta_micro, "
        "d.created_at, o.kind AS outcome_kind, o.carrier_reason "
        "FROM decisions d LEFT JOIN outcomes o "
        "ON o.merchant_id = d.merchant_id AND o.order_id = d.order_id "
        "WHERE d.merchant_id = :merchant_id")
    params = tenant.scope()
    if since:
        sql += " AND d.created_at >= :since"
        params["since"] = since
    rows = tenant.db.all(sql + " ORDER BY ABS(d.expected_delta_micro) DESC, d.created_at LIMIT 10",
                         params)
    return [{
        "order_id": str(row["order_id"]),
        "action": str(row["action"]),
        "reason": str(row["reason"]),
        "expected_delta_micro": int(row["expected_delta_micro"]),
        "outcome": row["outcome_kind"],
        "carrier_reason": str(row["carrier_reason"] or ""),
        "created_at": str(row["created_at"]),
    } for row in rows]


__all__ = ["REPORT_VERSION", "roi_report"]
