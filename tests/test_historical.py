"""Merchant historical CSV onboarding must be strict about input and honest about output."""

import pytest

from precourier.config import kolam_config
from precourier.domain import Mode
from precourier.historical import HistoricalImportError, parse_csv, project
from precourier.runtime import CountingIds, Runtime, SimClock
from precourier.store.db import Database
from precourier.tenancy import provision_merchant


def _tenant():
    db = Database.connect(":memory:")
    runtime = Runtime(clock=SimClock("2026-03-01T09:00:00Z"), ids=CountingIds())
    tenant = provision_merchant(
        db, runtime, "Kolam Ceramics", merchant_id="kolam_ceramics",
        config=kolam_config().replace(mode=Mode.ADVISORY))
    return tenant


CSV = """external_id,value,value_paise,outcome,phone,line1,locality,city,state,pincode,placed_at
KC-1,1200,,rto,9876543210,near temple,Koramangala,Bengaluru,Karnataka,560034,2026-02-01T09:00:00Z
KC-2,,150000,delivered,9876543210,Flat 402 Residency,Koramangala,Bengaluru,Karnataka,560034,2026-02-02T09:00:00Z
"""


def test_csv_parser_accepts_rupees_or_paise_and_normalizes_timestamps():
    rows = parse_csv(CSV)
    assert len(rows) == 2
    assert rows[0].value_paise == 120_000
    assert rows[1].value_paise == 150_000
    assert rows[0].outcome.value == "rto"
    assert rows[0].placed_at == "2026-02-01T09:00:00Z"


@pytest.mark.parametrize("content, message", [
    ("value,outcome\n1200,rto\n", "external_id"),
    ("external_id,outcome\nKC-1,rto\n", "value column"),
    ("external_id,value,outcome\nKC-1,1200,unknown\n", "outcome"),
    ("external_id,value,value_paise,outcome\nKC-1,12,1200,rto\n", "exactly one"),
])
def test_csv_parser_rejects_ambiguous_or_invalid_exports(content, message):
    with pytest.raises(HistoricalImportError, match=message):
        parse_csv(content)


def test_projection_separates_observed_ledger_from_assumed_uplift():
    report = project(_tenant(), parse_csv(CSV))
    assert report["provenance"] == "merchant_historical_csv"
    assert report["synthetic_data"] is False
    assert report["observed_rto_rate_ppm"] == 500_000
    assert report["assumptions"]["modeled_uplift_is_not_incremental_causal_evidence"] is True
    assert report["modeled_net_paise"] >= report["observed_net_paise"]
