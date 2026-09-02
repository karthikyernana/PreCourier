"""Shared fixtures. Plain functions, so `python -m pytest` and a bare script agree.

Deliberately does not depend on pytest for the fixture bodies themselves: the
backtest must run with nothing installed (PRD section 9), and a helper that only
works under a test runner is a helper the zero-install path cannot reuse.
"""

from __future__ import annotations

from typing import Optional, Tuple

from precourier import tenancy
from precourier.config import MerchantConfig, kolam_config
from precourier.domain import Address
from precourier.runtime import Runtime
from precourier.store.db import Database
from precourier.tenancy import Tenant


def fresh_db() -> Database:
    """An in-memory database with the schema applied."""
    return Database.connect(":memory:")


def deterministic_runtime(start: str = "2026-03-01T09:00:00Z") -> Runtime:
    """Simulated clock, counting ids. Every test runs at a known instant."""
    return Runtime.deterministic(start)


def make_tenant(db: Optional[Database] = None,
                runtime: Optional[Runtime] = None, *,
                name: str = "Kolam Ceramics",
                merchant_id: str = "kolam_ceramics",
                config: Optional[MerchantConfig] = None,
                **config_overrides: object) -> Tenant:
    """Provision one merchant and return its `Tenant`."""
    db = db or fresh_db()
    runtime = runtime or deterministic_runtime()
    if config is None:
        config = kolam_config(merchant_id=merchant_id)
    if config_overrides:
        config = config.replace(**config_overrides)
    return tenancy.provision_merchant(db, runtime, name,
                                      merchant_id=merchant_id, config=config)


def two_tenants() -> Tuple[Tenant, Tenant]:
    """Two merchants sharing one database, with deliberately different freight.

    Different cost models, because the interesting isolation bug is not "A can
    read B's rows" -- it is "A's decisions were computed with B's thresholds",
    which only shows up when the two configs disagree.
    """
    db = fresh_db()
    runtime = deterministic_runtime()
    a = make_tenant(db, runtime, name="Kolam Ceramics",
                    merchant_id="kolam_ceramics")
    b = tenancy.provision_merchant(
        db, runtime, "Rival Traders", merchant_id="rival_traders",
        config=MerchantConfig(
            merchant_id="rival_traders",
            monthly_incentive_budget_paise=5_000_00,
            gross_margin_bps=1800,              # thinner margin
            packaging_paise=4000,               # heavier packaging
            forward_freight_paise=12000,        # bulkier goods
            return_freight_paise=15000,
            origin_pincode="110001"))
    return a, b


#: A well-formed Bengaluru address, used wherever the address itself is not the
#: thing under test. 560034 is a real pincode in the reference table.
GOOD_ADDRESS = Address(
    line1="Flat 402, Ashirvad Residency",
    line2="12th Main Road",
    landmark="Opposite Sony World Signal",
    locality="Koramangala 4th Block",
    city="Bengaluru", state="Karnataka", pincode="560034")

#: The same buyer's address as a defective order feed supplies it: no house
#: number, no locality, a pincode that does not match the city.
DEFECTIVE_ADDRESS = Address(
    raw="near the big temple, bangalore",
    line1="near the big temple", city="Bangalore", pincode="110001")
