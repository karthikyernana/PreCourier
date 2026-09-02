"""Provision one local test merchant for inspecting the API and console.

This is developer setup only. It is not an application route and does not select a
simulated outbound provider. The production app remains fail-safe with no channel.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from precourier.config import kolam_config
from precourier.domain import Mode
from precourier.runtime import Runtime
from precourier.store.db import Database
from precourier.tenancy import issue_api_key, provision_merchant


def main() -> int:
    database_path = os.environ.get("PRECOURIER_DB", "var/precourier-local.db")
    pepper = os.environ.get("PRECOURIER_PEPPER")
    if not pepper:
        raise SystemExit(
            "PRECOURIER_PEPPER is required; set a local development secret before provisioning")

    db = Database.connect(database_path)
    runtime = Runtime.live(pepper=pepper)
    merchant_id = "local_merchant"
    if db.exists("SELECT 1 FROM merchants WHERE id = :id", {"id": merchant_id}):
        raise SystemExit(
            "%s already exists; choose a new PRECOURIER_DB for a clean local run"
            % merchant_id)
    tenant = provision_merchant(
        db, runtime, "Local Merchant", merchant_id=merchant_id,
        config=kolam_config(merchant_id=merchant_id).replace(mode=Mode.GATEWAY))
    key, record = issue_api_key(tenant, mode="test", label="local inspection")
    print("PRECOURIER_API_KEY=%s" % key)
    print("API_KEY_PREFIX=%s" % record.key_prefix)
    print("CONSOLE_URL=http://127.0.0.1:8000/console")
    print("DOCS_URL=http://127.0.0.1:8000/docs")
    db.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
