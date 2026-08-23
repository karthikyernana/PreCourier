"""The audit log: a hash chain per merchant.

Every consequential act -- a score, a decision, a message, a hold, a release, a
cancellation, a config change -- appends one row. Each row's hash covers the
previous row's hash, so the log is tamper-evident: altering or removing an entry
breaks every hash after it, and the break is localised to the row where it
happened.

    hash[n] = sha256( hash[n-1] || canonical_json(envelope[n]) )

The envelope covers the whole row, not just the payload. An early version hashed
only `payload_json`, which meant `merchant_id` or `created_at` could be edited
without breaking the chain -- a tamper-evident log that is not evident about
re-attributing an entry to a different tenant is not much of one.

**Chained per merchant.** A single global chain would be simpler, but a merchant
exporting their own audit trail could then not verify it: the verification needs
every intermediate hash, including other tenants' rows. Per-merchant chains make
a tenant-scoped export independently verifiable, which is the point of having one.

Appending is not optional and not conditional. PRD section 9: "an unexplained
decision is indistinguishable from a bug."
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from . import canonical
from .runtime import Runtime
from .store.db import Database

#: The chain's fixed root. Sixty-four zeros, so the first row's `prev_hash` is
#: distinguishable from a hash and from an empty string in a dump.
GENESIS = "0" * 64


# ==========================================================================
# what gets logged
# ==========================================================================

class AuditKind:
    """The vocabulary of auditable events.

    A closed set of string constants rather than an enum: the log is append-only
    and outlives the code, so a kind must remain readable after it is removed
    from this class. Adding one is a one-line change; renaming one is forbidden,
    because it would make old rows unattributable.
    """

    MERCHANT_CREATED = "merchant.created"
    CONFIG_CHANGED = "config.changed"
    API_KEY_ISSUED = "api_key.issued"
    API_KEY_REVOKED = "api_key.revoked"

    ORDER_RECEIVED = "order.received"
    ORDER_STATE_CHANGED = "order.state_changed"
    ORDER_HELD = "order.held"
    ORDER_RELEASED = "order.released"
    ORDER_AUTO_RELEASED = "order.auto_released"

    FEATURES_BUILT = "features.built"
    RISK_SCORED = "risk.scored"
    DIAGNOSED = "diagnosis.made"
    DECISION_MADE = "decision.made"
    DECISION_SUPPRESSED = "decision.suppressed"

    #: An internal error released an order rather than holding it. The most
    #: important kind in this list: it is the only evidence that a parcel shipped
    #: without a decision, and the "Where I was wrong" report counts these
    #: separately from orders the agent chose to leave alone.
    DECISION_FAILED_OPEN = "decision.failed_open"

    ACTION_ATTEMPTED = "action.attempted"
    ACTION_BLOCKED = "action.blocked"
    ADDRESS_PATCHED = "address.patched"
    ADDRESS_PATCH_REJECTED = "address.patch_rejected"
    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"

    LLM_PROPOSED = "llm.proposed"
    LLM_REJECTED = "llm.rejected"

    PAYMENT_LINK_CREATED = "payment.link_created"
    PAYMENT_CAPTURED = "payment.captured"

    OUTCOME_REPORTED = "outcome.reported"
    LEDGER_POSTED = "ledger.posted"

    MODEL_TRAINED = "model.trained"
    MODEL_PROMOTED = "model.promoted"
    MODEL_REJECTED = "model.rejected"

    WEBHOOK_SENT = "webhook.sent"
    OVERRIDE_BY_MERCHANT = "override.by_merchant"


@dataclasses.dataclass(frozen=True)
class AuditRecord:
    """One appended entry, as stored."""

    seq: int
    merchant_id: str
    merchant_seq: int
    kind: str
    actor: str
    order_id: Optional[str]
    payload: Dict[str, Any]
    prev_hash: str
    hash: str
    created_at: str


@dataclasses.dataclass(frozen=True)
class ChainVerification:
    """The result of re-walking a merchant's chain.

    `first_bad_seq` is the actionable part: it says which entry was altered, so
    an investigation starts at a row rather than at a table.
    """

    merchant_id: str
    ok: bool
    entries_checked: int
    first_bad_seq: Optional[int] = None
    detail: str = ""

    def summary(self) -> str:
        if self.ok:
            return "%s: %d entries, chain intact" % (
                self.merchant_id, self.entries_checked)
        return "%s: BROKEN at merchant_seq %s -- %s" % (
            self.merchant_id, self.first_bad_seq, self.detail)


# ==========================================================================
# the log
# ==========================================================================

def envelope(*, merchant_id: str, merchant_seq: int, kind: str, actor: str,
             order_id: Optional[str], created_at: str,
             payload: Dict[str, Any]) -> Dict[str, Any]:
    """The exact structure that gets hashed.

    A free function, so verification builds it the same way appending did. If
    this were a method on the log, a future refactor could change the append path
    and leave the verify path building a different envelope -- and the chain would
    then report tampering that never happened.
    """
    return {
        "actor": actor,
        "created_at": created_at,
        "kind": kind,
        "merchant_id": merchant_id,
        "merchant_seq": merchant_seq,
        "order_id": order_id,
        "payload": payload,
    }


def compute_hash(prev_hash: str, env: Dict[str, Any]) -> str:
    return canonical.sha256_hex(prev_hash.encode("ascii"), b"\x00",
                                canonical.dumpb(env))


class AuditLog:
    """Append-only, hash-chained, tenant-scoped.

    Writes go through `Database.tx`, so an audit row and the state change it
    describes commit together or not at all. A decision recorded without its
    audit entry, or an audit entry without its decision, would both be worse than
    a failed request.
    """

    def __init__(self, db: Database, runtime: Runtime) -> None:
        self._db = db
        self._rt = runtime

    def append(self, merchant_id: str, kind: str, payload: Dict[str, Any], *,
               order_id: Optional[str] = None,
               actor: str = "agent") -> AuditRecord:
        """Append one entry and return it.

        `payload` must be canonical-JSON-safe: integers, strings, bools, None,
        and containers of those. A float raises `canonical.NotCanonical` rather
        than being coerced, because money is integer paise everywhere else and a
        float arriving here means something upstream lost precision.
        """
        canonical.dumps(payload)          # fail before touching the database
        created_at = self._rt.stamp()

        with self._db.tx():
            tail = self._db.one(
                "SELECT merchant_seq, hash FROM audit_log "
                "WHERE merchant_id = :m ORDER BY merchant_seq DESC LIMIT 1",
                {"m": merchant_id})
            if tail is None:
                merchant_seq, prev_hash = 1, GENESIS
            else:
                merchant_seq = int(tail["merchant_seq"]) + 1
                prev_hash = str(tail["hash"])

            env = envelope(merchant_id=merchant_id, merchant_seq=merchant_seq,
                           kind=kind, actor=actor, order_id=order_id,
                           created_at=created_at, payload=payload)
            digest = compute_hash(prev_hash, env)

            self._db.insert("audit_log", {
                "merchant_id": merchant_id,
                "merchant_seq": merchant_seq,
                "order_id": order_id,
                "kind": kind,
                "actor": actor,
                "payload_json": canonical.dumps(payload),
                "prev_hash": prev_hash,
                "hash": digest,
                "created_at": created_at,
            })
            seq = int(self._db.scalar(
                "SELECT seq FROM audit_log WHERE merchant_id = :m "
                "AND merchant_seq = :n", {"m": merchant_id, "n": merchant_seq},
                default=0))

        return AuditRecord(seq=seq, merchant_id=merchant_id,
                           merchant_seq=merchant_seq, kind=kind, actor=actor,
                           order_id=order_id, payload=payload,
                           prev_hash=prev_hash, hash=digest,
                           created_at=created_at)

    # ------------------------------------------------------------------ reads

    def verify(self, merchant_id: str) -> ChainVerification:
        """Re-walk the chain from genesis, recomputing every hash.

        Catches an edited payload, a re-attributed entry, a deleted row (the
        sequence gaps), and a re-ordered one. Does not catch truncation of the
        tail -- no self-contained log can, because a shorter valid chain is still
        a valid chain. The countermeasure is publishing the head hash; the console
        shows it and `precourier.reporting` includes it in an export.
        """
        rows = self._db.all(
            "SELECT merchant_seq, kind, actor, order_id, payload_json, "
            "       prev_hash, hash, created_at "
            "FROM audit_log WHERE merchant_id = :m ORDER BY merchant_seq",
            {"m": merchant_id})

        expected_prev = GENESIS
        for index, row in enumerate(rows, start=1):
            seq = int(row["merchant_seq"])
            if seq != index:
                return ChainVerification(
                    merchant_id, False, index - 1, seq,
                    "sequence jumped to %d at position %d: an entry was removed"
                    % (seq, index))
            if str(row["prev_hash"]) != expected_prev:
                return ChainVerification(
                    merchant_id, False, index - 1, seq,
                    "prev_hash does not match the preceding entry's hash")

            env = envelope(
                merchant_id=merchant_id, merchant_seq=seq,
                kind=str(row["kind"]), actor=str(row["actor"]),
                order_id=row["order_id"], created_at=str(row["created_at"]),
                payload=canonical.loads(str(row["payload_json"])))
            recomputed = compute_hash(expected_prev, env)
            if recomputed != str(row["hash"]):
                return ChainVerification(
                    merchant_id, False, index - 1, seq,
                    "stored hash does not match the entry's contents")
            expected_prev = recomputed

        return ChainVerification(merchant_id, True, len(rows))

    def head(self, merchant_id: str) -> str:
        """The chain's current tip. Publish this and truncation becomes visible."""
        return str(self._db.scalar(
            "SELECT hash FROM audit_log WHERE merchant_id = :m "
            "ORDER BY merchant_seq DESC LIMIT 1",
            {"m": merchant_id}, default=GENESIS))

    def for_order(self, merchant_id: str, order_id: str) -> List[AuditRecord]:
        """One order's history, in order. This is the console's timeline."""
        rows = self._db.all(
            "SELECT * FROM audit_log WHERE merchant_id = :m AND order_id = :o "
            "ORDER BY merchant_seq", {"m": merchant_id, "o": order_id})
        return [_record(row) for row in rows]

    def recent(self, merchant_id: str, limit: int = 100,
               kind: Optional[str] = None) -> List[AuditRecord]:
        sql = ("SELECT * FROM audit_log WHERE merchant_id = :m"
               + (" AND kind = :k" if kind else "")
               + " ORDER BY merchant_seq DESC LIMIT :n")
        params: Dict[str, Any] = {"m": merchant_id, "n": int(limit)}
        if kind:
            params["k"] = kind
        return [_record(row) for row in self._db.all(sql, params)]

    def export(self, merchant_id: str) -> Dict[str, Any]:
        """A self-contained, independently verifiable export of one tenant's log.

        Includes `genesis` and `head` so a recipient can re-run `verify`'s
        arithmetic without this codebase -- the chain is plain SHA-256 over
        canonical JSON, which is reimplementable in an afternoon in any language.
        That is deliberate: an audit trail only a vendor can check is a promise,
        not a proof.
        """
        records = [dataclasses.asdict(r) for r in self._chronological(merchant_id)]
        return {
            "merchant_id": merchant_id,
            "algorithm": "sha256(prev_hash || 0x00 || canonical_json(envelope))",
            "canonical_json": "sorted keys, separators (',',':'), no floats",
            "genesis": GENESIS,
            "head": self.head(merchant_id),
            "entries": records,
        }

    def _chronological(self, merchant_id: str) -> List[AuditRecord]:
        rows = self._db.all(
            "SELECT * FROM audit_log WHERE merchant_id = :m "
            "ORDER BY merchant_seq", {"m": merchant_id})
        return [_record(row) for row in rows]


def _record(row: Any) -> AuditRecord:
    return AuditRecord(
        seq=int(row["seq"]), merchant_id=str(row["merchant_id"]),
        merchant_seq=int(row["merchant_seq"]), kind=str(row["kind"]),
        actor=str(row["actor"]), order_id=row["order_id"],
        payload=canonical.loads(str(row["payload_json"])),
        prev_hash=str(row["prev_hash"]), hash=str(row["hash"]),
        created_at=str(row["created_at"]))
