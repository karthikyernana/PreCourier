"""Tenancy: merchants, API keys, and the scoping handle every query needs.

The isolation guarantee this module exists to provide is narrow and absolute: a
request authenticated as merchant A cannot observe or affect merchant B's rows.
PRD section 14 requires it be "enforced at the query layer, not the view layer,
and asserted by test".

The mechanism is `Tenant`. It is the only way to reach a repository, it carries a
`merchant_id` that came from a key hash rather than from a request parameter, and
its `scope()` seeds every query's parameters with that id. Repositories are then
written so their SQL always filters on `:merchant_id` -- and because that is a
convention a tired author can forget, `tests/test_tenancy.py` parses every SQL
string in `repo.py` and fails on any statement touching a tenant-scoped table
without it. A structural test, not a spot check.

Keys are stored only as a hash, so a stolen database does not yield working
credentials. The plaintext is returned exactly once, from `issue_api_key`, and is
never written to a row, a log line, or a template.
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from typing import Any, Dict, List, Optional, Tuple

from . import canonical
from .audit import AuditKind, AuditLog
from .config import DEFAULT_ACTION_ECONOMICS, ActionEconomics, MerchantConfig
from .domain import ActionKind, Channel, Mode
from .runtime import Runtime, key_prefix, mint_api_key
from .store.db import Database, StorageError


class AuthError(Exception):
    """Authentication failed. Never says why -- see `authenticate`."""


class TenantError(StorageError):
    """A tenancy operation that cannot be satisfied."""


#: Config columns, derived from the dataclass so adding a field to
#: `MerchantConfig` cannot silently fail to persist. `action_economics` lives in
#: its own table and is handled separately.
_CONFIG_COLUMNS: Tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(MerchantConfig)
    if f.name not in ("merchant_id", "action_economics"))

#: The declared type of each config column, resolved through `get_type_hints`
#: because `from __future__ import annotations` leaves `field.type` as a string.
#:
#: Persistence is driven from these rather than from a hand-written list of column
#: names. Adding a field to `MerchantConfig` previously meant editing three
#: functions in agreement -- the writer, the reader and the audit payload -- and
#: two of the three failing to notice is exactly the shape of bug that quietly
#: drops a merchant's setting on save.
_CONFIG_TYPES: Dict[str, Any] = {
    name: hint for name, hint in typing.get_type_hints(MerchantConfig).items()
    if name in _CONFIG_COLUMNS}


def _to_column(value: Any) -> Any:
    """Flatten one config value to something SQLite and canonical JSON accept."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, bool):
        return int(value)
    return value


def _from_column(name: str, value: Any) -> Any:
    """Rebuild one config value from its stored form, per the declared type."""
    hint = _CONFIG_TYPES[name]
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return hint(str(value))
    # bool before int: `issubclass(bool, int)` is true, so the order matters and
    # getting it wrong would store every flag as 0 or 1 and hand back an int.
    if hint is bool:
        return bool(value)
    if hint is str:
        return str(value)
    return int(value)



# ==========================================================================
# the scoping handle
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class Tenant:
    """An authenticated merchant, and everything a request needs to act as one.

    Construct via `authenticate` or `provision_merchant`. Constructing one by
    hand from a `merchant_id` taken off a request would defeat the whole point,
    so nothing in the API layer does -- the id always originates from a key hash
    or from a console session.

    `mode` is carried separately from `config.mode` because a single order can be
    forced into shadow mode for a canary without rewriting the merchant's config.
    """

    merchant_id: str
    name: str
    db: Database
    runtime: Runtime
    config: MerchantConfig
    audit: AuditLog
    key_mode: str = "test"

    @property
    def is_live(self) -> bool:
        return self.key_mode == "live"

    @property
    def mode(self) -> Mode:
        return self.config.mode

    @property
    def config_revision(self) -> int:
        """Which revision of the cost model this tenant's config is.

        Read from the database on each call rather than carried on the handle. A
        cached revision on a long-lived `Tenant` would go stale the moment the
        merchant edited their freight in the console, and the number's only job is
        to be stamped onto `decisions.config_revision` so a decision can be
        re-explained against the arithmetic that actually produced it. A stale
        stamp there is worse than no stamp: it would attribute a decision to
        numbers the merchant had already replaced.

        One primary-key lookup. The name below is the module-level function of the
        same name, not this property -- a method body resolves bare names to
        globals, never to class attributes.
        """
        return config_revision(self.db, self.merchant_id)

    def scope(self, **params: Any) -> Dict[str, Any]:
        """Query parameters pre-seeded with this tenant's id.

        Every repository method starts from this. A caller-supplied
        `merchant_id` is refused rather than overwritten: silently ignoring one
        would hide the bug that produced it.
        """
        if "merchant_id" in params:
            raise TenantError(
                "merchant_id must come from the authenticated Tenant, not from "
                "the caller -- passing one here is how cross-tenant reads happen")
        params["merchant_id"] = self.merchant_id
        return params

    def with_config(self, config: MerchantConfig) -> "Tenant":
        return dataclasses.replace(self, config=config)


# ==========================================================================
# provisioning
# ==========================================================================

def provision_merchant(db: Database, runtime: Runtime, name: str, *,
                       merchant_id: Optional[str] = None,
                       config: Optional[MerchantConfig] = None,
                       **config_kwargs: Any) -> Tenant:
    """Create a merchant, its config, and its action economics, atomically.

    Either pass a fully-built `config` or pass keyword overrides and let this
    build one. Whichever path, `validate()` runs before anything is written: a
    merchant with negative freight should fail at onboarding, not at the first
    decision.
    """
    merchant_id = merchant_id or runtime.new_id("merchant")

    if config is None:
        kwargs = dict(config_kwargs)
        kwargs.setdefault("monthly_incentive_budget_paise", 0)
        config = MerchantConfig(merchant_id=merchant_id, **kwargs)
    elif config.merchant_id != merchant_id:
        config = config.replace(merchant_id=merchant_id)
    config = config.validate()

    audit = AuditLog(db, runtime)
    now = runtime.stamp()

    with db.tx():
        if db.exists("SELECT 1 FROM merchants WHERE id = :id",
                     {"id": merchant_id}):
            raise TenantError("merchant %s already exists" % merchant_id)
        db.insert("merchants", {"id": merchant_id, "name": name,
                                "status": "active", "created_at": now})
        _write_config(db, config, revision=1, now=now)
        audit.append(merchant_id, AuditKind.MERCHANT_CREATED,
                     {"name": name, "mode": config.mode.value,
                      "config": _config_payload(config)},
                     actor="system")

    return Tenant(merchant_id=merchant_id, name=name, db=db, runtime=runtime,
                  config=config, audit=audit)


def load_tenant(db: Database, runtime: Runtime, merchant_id: str, *,
                key_mode: str = "test") -> Tenant:
    """Load a tenant by id. For the console and for internal jobs.

    Not reachable from an API request -- those go through `authenticate`, which
    derives the id from a key hash. A console session supplies the id from a
    signed cookie, which is the same trust chain by a different route.
    """
    row = db.one("SELECT id, name, status FROM merchants WHERE id = :id",
                 {"id": merchant_id})
    if row is None:
        raise TenantError("no merchant %s" % merchant_id)
    if str(row["status"]) != "active":
        raise AuthError("merchant is %s" % row["status"])
    return Tenant(merchant_id=merchant_id, name=str(row["name"]), db=db,
                  runtime=runtime, config=load_config(db, merchant_id),
                  audit=AuditLog(db, runtime), key_mode=key_mode)


def list_merchants(db: Database) -> List[Dict[str, Any]]:
    """Every merchant. Operator-level, never exposed to a tenant-scoped route."""
    return [dict(r) for r in db.all(
        "SELECT id, name, status, created_at FROM merchants ORDER BY created_at")]


# ==========================================================================
# configuration
# ==========================================================================

def load_config(db: Database, merchant_id: str) -> MerchantConfig:
    row = db.one("SELECT * FROM merchant_config WHERE merchant_id = :m",
                 {"m": merchant_id})
    if row is None:
        raise TenantError("no config for merchant %s" % merchant_id)

    kwargs: Dict[str, Any] = {"merchant_id": merchant_id}
    for column in _CONFIG_COLUMNS:
        kwargs[column] = _from_column(column, row[column])

    economics = dict(DEFAULT_ACTION_ECONOMICS)
    for erow in db.all("SELECT * FROM merchant_action_economics "
                       "WHERE merchant_id = :m", {"m": merchant_id}):
        try:
            action = ActionKind(str(erow["action"]))
        except ValueError:
            # A row for an action this build no longer knows. Skipped rather than
            # fatal: the table outlives the code, same reasoning as AuditKind.
            continue
        economics[action] = ActionEconomics(
            cost_paise=int(erow["cost_paise"]),
            harm_bps=int(erow["harm_bps"]),
            reversible=bool(erow["reversible"]),
            contacts_buyer=bool(erow["contacts_buyer"]))
    kwargs["action_economics"] = economics

    return MerchantConfig(**kwargs)


def save_config(tenant: Tenant, config: MerchantConfig, *,
                actor: str = "merchant") -> Tenant:
    """Persist a config change, bump the revision, and record the diff.

    The revision matters downstream: `decisions.config_revision` records which
    cost model produced a decision, so a merchant who halves their freight in
    March cannot make February's decisions look wrong when the console
    re-renders them.
    """
    if config.merchant_id != tenant.merchant_id:
        raise TenantError("config belongs to %s, tenant is %s"
                          % (config.merchant_id, tenant.merchant_id))
    config = config.validate()
    now = tenant.runtime.stamp()

    with tenant.db.tx():
        current = int(tenant.db.scalar(
            "SELECT revision FROM merchant_config WHERE merchant_id = :m",
            {"m": tenant.merchant_id}, default=0))
        revision = current + 1
        before = _config_payload(tenant.config)
        after = _config_payload(config)
        changed = {k: [before.get(k), after[k]] for k in after
                   if before.get(k) != after[k]}

        _write_config(tenant.db, config, revision=revision, now=now)
        tenant.db.insert("merchant_config_history", {
            "id": tenant.runtime.new_id("config"),
            "merchant_id": tenant.merchant_id,
            "revision": revision,
            "payload_json": canonical.dumps(after),
            "created_at": now})
        tenant.audit.append(tenant.merchant_id, AuditKind.CONFIG_CHANGED,
                            {"revision": revision, "changed": changed},
                            actor=actor)

    return tenant.with_config(config)


def config_revision(db: Database, merchant_id: str) -> int:
    return int(db.scalar(
        "SELECT revision FROM merchant_config WHERE merchant_id = :m",
        {"m": merchant_id}, default=1))


def _config_payload(config: MerchantConfig) -> Dict[str, Any]:
    """Canonical-JSON-safe view of a config, for the audit log and history.

    Enum and bool are flattened to primitives; `action_economics` is keyed by the
    action's string value so the payload is stable across a Python version change
    that alters enum iteration or repr.
    """
    payload: Dict[str, Any] = {}
    for column in _CONFIG_COLUMNS:
        payload[column] = _to_column(getattr(config, column))
    payload["action_economics"] = {
        action.value: {"cost_paise": econ.cost_paise,
                       "harm_bps": econ.harm_bps,
                       "reversible": int(econ.reversible),
                       "contacts_buyer": int(econ.contacts_buyer)}
        for action, econ in sorted(config.action_economics.items(),
                                   key=lambda kv: kv[0].value)}
    return payload


def _write_config(db: Database, config: MerchantConfig, *, revision: int,
                  now: str) -> None:
    """Upsert `merchant_config` and replace `merchant_action_economics`.

    Called inside a caller-owned transaction. The economics rows are deleted and
    rewritten rather than upserted one by one, because a dropped *override* must
    actually be dropped -- an upsert would leave the merchant's old custom price
    in place, and `load_config` would keep applying it.

    Dropping an override reverts that action to `DEFAULT_ACTION_ECONOMICS`; it
    does not make the action unpriced. See `MerchantConfig.economics` for why an
    unpriced action would be worse than a mispriced one.
    """
    values: Dict[str, Any] = {"merchant_id": config.merchant_id,
                              "revision": revision, "updated_at": now}
    for column in _CONFIG_COLUMNS:
        values[column] = _to_column(getattr(config, column))

    columns = sorted(values)
    db.execute(
        "INSERT INTO merchant_config (%s) VALUES (%s) "
        "ON CONFLICT(merchant_id) DO UPDATE SET %s" % (
            ", ".join(columns),
            ", ".join(":" + c for c in columns),
            ", ".join("%s = excluded.%s" % (c, c)
                      for c in columns if c != "merchant_id")),
        values)

    db.execute("DELETE FROM merchant_action_economics WHERE merchant_id = :m",
               {"m": config.merchant_id})
    db.executemany(
        "INSERT INTO merchant_action_economics "
        "(merchant_id, action, cost_paise, harm_bps, reversible, "
        " contacts_buyer, updated_at) "
        "VALUES (:merchant_id, :action, :cost_paise, :harm_bps, :reversible, "
        "        :contacts_buyer, :updated_at)",
        [{"merchant_id": config.merchant_id, "action": action.value,
          "cost_paise": econ.cost_paise, "harm_bps": econ.harm_bps,
          "reversible": int(econ.reversible),
          "contacts_buyer": int(econ.contacts_buyer), "updated_at": now}
         for action, econ in sorted(config.action_economics.items(),
                                    key=lambda kv: kv[0].value)])


# ==========================================================================
# API keys
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class ApiKeyRecord:
    """A key as stored -- which is to say, without the key."""

    id: str
    merchant_id: str
    key_prefix: str
    mode: str
    label: str
    created_at: str
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.revoked_at is None


def issue_api_key(tenant: Tenant, *, mode: str = "test",
                  label: str = "") -> Tuple[str, ApiKeyRecord]:
    """Mint a key. Returns `(plaintext, record)`.

    The plaintext is the only copy that will ever exist -- it is hashed on the
    way in and not stored. The caller shows it once and discards it; the console
    displays it on the creation screen and never again.
    """
    plaintext = mint_api_key(mode)
    now = tenant.runtime.stamp()
    record = ApiKeyRecord(
        id=tenant.runtime.new_id("api_key"), merchant_id=tenant.merchant_id,
        key_prefix=key_prefix(plaintext), mode=mode, label=label,
        created_at=now)

    with tenant.db.tx():
        tenant.db.insert("api_keys", {
            "id": record.id, "merchant_id": record.merchant_id,
            "key_hash": tenant.runtime.hash_secret(plaintext),
            "key_prefix": record.key_prefix, "mode": mode, "label": label,
            "created_at": now})
        # `key_prefix` is safe to log; the plaintext is not, and is absent here.
        tenant.audit.append(tenant.merchant_id, AuditKind.API_KEY_ISSUED,
                            {"api_key_id": record.id, "mode": mode,
                             "label": label, "key_prefix": record.key_prefix},
                            actor="merchant")
    return plaintext, record


def revoke_api_key(tenant: Tenant, api_key_id: str) -> None:
    now = tenant.runtime.stamp()
    with tenant.db.tx():
        cursor = tenant.db.execute(
            "UPDATE api_keys SET revoked_at = :now "
            "WHERE id = :id AND merchant_id = :merchant_id "
            "AND revoked_at IS NULL",
            tenant.scope(id=api_key_id, now=now))
        if cursor.rowcount == 0:
            # Scoped by merchant_id, so this covers "not yours" and "not found"
            # with one message -- a distinct error would confirm the existence of
            # another tenant's key id.
            raise TenantError("no active key %s for this merchant" % api_key_id)
        tenant.audit.append(tenant.merchant_id, AuditKind.API_KEY_REVOKED,
                            {"api_key_id": api_key_id}, actor="merchant")


def list_api_keys(tenant: Tenant) -> List[ApiKeyRecord]:
    rows = tenant.db.all(
        "SELECT id, merchant_id, key_prefix, mode, label, created_at, "
        "       last_used_at, revoked_at "
        "FROM api_keys WHERE merchant_id = :merchant_id "
        "ORDER BY created_at DESC", tenant.scope())
    return [ApiKeyRecord(
        id=str(r["id"]), merchant_id=str(r["merchant_id"]),
        key_prefix=str(r["key_prefix"]), mode=str(r["mode"]),
        label=str(r["label"]), created_at=str(r["created_at"]),
        last_used_at=r["last_used_at"], revoked_at=r["revoked_at"])
        for r in rows]


def authenticate(db: Database, runtime: Runtime, plaintext: str) -> Tenant:
    """Resolve an API key to a `Tenant`, or raise `AuthError`.

    Every failure -- malformed, unknown, revoked, suspended merchant -- raises
    the same bare `AuthError`. Distinguishing "no such key" from "revoked key"
    would let an attacker confirm that a key they found was once real.

    Lookup is by hash, so the plaintext is never compared against anything
    stored, and `last_used_at` is updated on success because a merchant needs to
    know whether a key they are about to revoke is still in use.
    """
    if not plaintext or not plaintext.startswith(("pk_test_", "pk_live_")):
        raise AuthError("invalid credentials")

    row = db.one(
        "SELECT k.id, k.merchant_id, k.mode, k.revoked_at, "
        "       m.name AS merchant_name, m.status AS merchant_status "
        "FROM api_keys k JOIN merchants m ON m.id = k.merchant_id "
        "WHERE k.key_hash = :h "
        "/* tenancy-exempt: this query *discovers* the tenant. There is no "
        "   merchant_id to filter on yet -- the key hash is the only input, and "
        "   it is what the returned merchant_id is derived from. */",
        {"h": runtime.hash_secret(plaintext)})
    if row is None or row["revoked_at"] is not None:
        raise AuthError("invalid credentials")
    if str(row["merchant_status"]) != "active":
        raise AuthError("invalid credentials")

    merchant_id = str(row["merchant_id"])
    db.execute("UPDATE api_keys SET last_used_at = :now WHERE id = :id "
               "/* tenancy-exempt: id came from the lookup above, which already "
               "   resolved the owning merchant. */",
               {"now": runtime.stamp(), "id": str(row["id"])})

    return Tenant(merchant_id=merchant_id, name=str(row["merchant_name"]),
                  db=db, runtime=runtime, config=load_config(db, merchant_id),
                  audit=AuditLog(db, runtime), key_mode=str(row["mode"]))
