"""The controlled experiment must be honest, reproducible, and reportable."""

import json

from precourier.backtest import REPORT_VERSION, multi_seed, run


def test_backtest_is_byte_stable_and_discloses_synthetic_provenance():
    first = json.dumps(run(42, n_orders=40), sort_keys=True,
                       separators=(",", ":"))
    second = json.dumps(run(42, n_orders=40), sort_keys=True,
                        separators=(",", ":"))
    assert first == second
    report = json.loads(first)
    assert report["report_version"] == REPORT_VERSION
    assert report["synthetic_data"] is True
    assert report["provenance"] == "synthetic_world"
    assert set(report["arms"]) == {"ship_all", "block_best", "agent"}


def test_backtest_runs_all_declared_sensitivity_cells():
    report = run(7, n_orders=20, include_sensitivity=True)
    assert len(report["sensitivity"]) == 9
    assert all(row["agent_net_per_order_paise"] is not None
               for row in report["sensitivity"])


def test_backtest_can_aggregate_fixed_seeds():
    report = multi_seed(n_orders=10, seeds=(1, 2, 3))
    assert report["seeds"] == [1, 2, 3]
    assert len(report["agent_net_per_order_paise"]) == 3
    assert report["mean_net_per_order_paise"] == sum(
        report["agent_net_per_order_paise"]) // 3
