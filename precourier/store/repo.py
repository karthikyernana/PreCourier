"""Repositories: the only module in the package that writes SQL.

Two rules hold everywhere below, and both exist because of a specific failure they
prevent.

**Every method takes a `Tenant`, never a `merchant_id`.** The id then always
originates from an authenticated key hash rather than from a request field, and
`tenant.scope()` seeds it into the parameters. `tests/test_tenancy.py` parses every
SQL literal in this file and fails any statement that touches a tenant-scoped
table without filtering on `:merchant_id` -- so the guarantee is structural, not a
habit that survives only while someone is paying attention.

**State changes and their audit entries commit together.** Each mutating method
opens `db.tx()` and appends to the audit log inside it. A decision recorded
without its audit row would leave the console unable to explain an action the
merchant can see happened, which PRD section 9 treats as indistinguishable from a
bug.

Reads return domain objects, not rows. Nothing above this module knows a column
name, so the Postgres port stays inside `store/`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import canonical
from ..audit import AuditKind
from ..domain import (ActionKind, Address, Buyer, Channel, Diagnosis,
                      Evidence, IllegalTransition,
                      LedgerReason, Mode, Order, OrderState, OUTCOME_STATE,
                      Outcome, OutcomeKind, PAID_ACTIONS, TERMINAL_STATES,
                      TRANSITIONS, can_transition, path_to)
from .. import ledger as ledger_module
from ..ledger import LedgerBreakdown
from ..money import Paise
from ..pii import normalise_phone, phone_hash
from ..tenancy import Tenant
from .db import StorageError


class NotFound(StorageError):
    """A row this tenant may see does not exist.

    One error for "absent" and "another tenant's", deliberately: distinguishing
    them would turn a 404 into an existence oracle for other merchants' ids.
    """


class Conflict(StorageError):
    """A uniqueness or idempotency conflict the caller must resolve."""


class ReasoningRepo:
    """Append-only persistence for observable reasoning artifacts."""

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def record(self, order_id: str, *, payload: Dict[str, Any],
               risk_score: Any, diagnosis: Diagnosis) -> Tuple[str, str, str]:
        feature_id = self._t.runtime.new_id("features")
        risk_id = self._t.runtime.new_id("risk")
        diagnosis_id = self._t.runtime.new_id("diagnosis")
        now = self._t.runtime.stamp()
        evidence = [{"code": item.code, "detail": item.detail,
                     "weight": item.weight} for item in diagnosis.evidence]
        with self._db.tx():
            self._db.insert("features", {
                "id": feature_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "pass_no": 1,
                "payload_json": canonical.dumps(payload), "created_at": now})
            self._t.audit.append(
                self._t.merchant_id, AuditKind.FEATURES_BUILT,
                {"order_id": order_id, "features_id": feature_id,
                 "pass_no": 1}, order_id=order_id)
            self._db.insert("risk_scores", {
                "id": risk_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "features_id": feature_id,
                "p_rto_ppm": int(risk_score.p_rto_ppm),
                "raw_ppm": int(risk_score.raw_ppm)
                if risk_score.raw_ppm is not None else None,
                "model_version": risk_score.model_version,
                "created_at": now})
            self._t.audit.append(
                self._t.merchant_id, AuditKind.RISK_SCORED,
                {"order_id": order_id, "risk_score_id": risk_id,
                 "p_rto_ppm": int(risk_score.p_rto_ppm),
                 "model_version": risk_score.model_version},
                order_id=order_id)
            self._db.insert("diagnoses", {
                "id": diagnosis_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "cause": diagnosis.cause.value,
                "runner_up": diagnosis.runner_up.value
                if diagnosis.runner_up is not None else None,
                "evidence_json": canonical.dumps(evidence),
                "detail": diagnosis.detail, "created_at": now})
            self._t.audit.append(
                self._t.merchant_id, AuditKind.DIAGNOSED,
                {"order_id": order_id, "diagnosis_id": diagnosis_id,
                 "cause": diagnosis.cause.value,
                 "evidence_count": len(evidence)}, order_id=order_id)
        return feature_id, risk_id, diagnosis_id

    def for_order(self, order_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Read immutable reasoning artifacts for an order detail view."""
        features = self._db.all(
            "SELECT id, pass_no, payload_json, created_at FROM features "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY pass_no", self._t.scope(o=order_id))
        scores = self._db.all(
            "SELECT id, features_id, p_rto_ppm, raw_ppm, model_version, created_at "
            "FROM risk_scores WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY created_at, id", self._t.scope(o=order_id))
        diagnoses = self._db.all(
            "SELECT id, cause, runner_up, evidence_json, detail, created_at "
            "FROM diagnoses WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY created_at, id", self._t.scope(o=order_id))
        return {
            "features": [{"id": str(row["id"]), "pass_no": int(row["pass_no"]),
                          "payload": canonical.loads(str(row["payload_json"])),
                          "created_at": str(row["created_at"])} for row in features],
            "risk_scores": [{"id": str(row["id"]),
                             "features_id": row["features_id"],
                             "p_rto_ppm": int(row["p_rto_ppm"]),
                             "raw_ppm": row["raw_ppm"],
                             "model_version": str(row["model_version"]),
                             "created_at": str(row["created_at"])} for row in scores],
            "diagnoses": [{"id": str(row["id"]), "cause": str(row["cause"]),
                           "runner_up": row["runner_up"],
                           "evidence": canonical.loads(str(row["evidence_json"])),
                           "detail": str(row["detail"]),
                           "created_at": str(row["created_at"])} for row in diagnoses],
        }


class MessageRepo:
    """Immutable inbound/outbound conversation records for the MVP channel."""

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def record_outbound(self, order_id: str, buyer_id: str, *, action_id: str,
                        channel: Channel, body: str, template_id: str = "",
                        provider_ref: str = "", status: str = "sent",
                        message_id: Optional[str] = None) -> str:
        return self._record(
            order_id, buyer_id, action_id=action_id, channel=channel,
            direction="outbound", body=body, template_id=template_id,
            provider_ref=provider_ref, status=status, message_id=message_id)

    def record_inbound(self, order_id: str, buyer_id: str, *, channel: Channel,
                       body: str, provider_ref: str = "",
                       message_id: Optional[str] = None) -> str:
        return self._record(
            order_id, buyer_id, action_id=None, channel=channel,
            direction="inbound", body=body, template_id="",
            provider_ref=provider_ref, status="received", message_id=message_id)

    def for_order(self, order_id: str) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._db.all(
            "SELECT id, order_id, buyer_id, action_id, channel, direction, "
            "template_id, body, provider_ref, status, created_at "
            "FROM messages WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY created_at, id", self._t.scope(o=order_id))]

    def _record(self, order_id: str, buyer_id: str, *, action_id: Optional[str],
                channel: Channel, direction: str, body: str,
                template_id: str, provider_ref: str, status: str,
                message_id: Optional[str]) -> str:
        message_id = message_id or self._t.runtime.new_id("message")
        now = self._t.runtime.stamp()
        with self._db.tx():
            self._db.insert("messages", {
                "id": message_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "buyer_id": buyer_id,
                "action_id": action_id, "channel": channel.value,
                "direction": direction, "template_id": template_id,
                "body": body, "provider_ref": provider_ref,
                "status": status, "created_at": now})
            self._t.audit.append(
                self._t.merchant_id,
                AuditKind.MESSAGE_SENT if direction == "outbound"
                else AuditKind.MESSAGE_RECEIVED,
                {"message_id": message_id, "order_id": order_id,
                 "buyer_id": buyer_id, "channel": channel.value,
                 "status": status, "provider_ref": provider_ref},
                order_id=order_id, actor="simulated" if channel is Channel.SIMULATED
                else "provider")
        return message_id


@dataclasses.dataclass(frozen=True)
class OutboxRecord:
    id: str
    order_id: str
    action_id: str
    kind: str
    payload: Dict[str, Any]
    idempotency_key: str
    status: str
    attempts: int
    provider_ref: str
    last_error: str
    available_at: str
    created_at: str
    updated_at: str


class OutboxRepo:
    """Durable provider-intent queue, always tenant-scoped."""

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def enqueue(self, order_id: str, action_id: str, kind: ActionKind,
                payload: Dict[str, Any], *, idempotency_key: str) -> OutboxRecord:
        now = self._t.runtime.stamp()
        outbox_id = self._t.runtime.new_id("delivery")
        with self._db.tx():
            existing = self._db.one(
                "SELECT * FROM action_outbox WHERE merchant_id = :merchant_id "
                "AND idempotency_key = :key",
                self._t.scope(key=idempotency_key))
            if existing is not None:
                return self._to_record(existing)
            self._db.insert("action_outbox", {
                "id": outbox_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "action_id": action_id,
                "kind": kind.value, "payload_json": canonical.dumps(payload),
                "idempotency_key": idempotency_key, "status": "pending",
                "attempts": 0, "provider_ref": "", "last_error": "",
                "available_at": now, "created_at": now, "updated_at": now})
            self._t.audit.append(
                self._t.merchant_id, "provider.intent_queued",
                {"outbox_id": outbox_id, "order_id": order_id,
                 "action_id": action_id, "kind": kind.value},
                order_id=order_id)
        return self.get(outbox_id)

    def get(self, outbox_id: str) -> OutboxRecord:
        row = self._db.one(
            "SELECT * FROM action_outbox WHERE merchant_id = :merchant_id AND id = :id",
            self._t.scope(id=outbox_id))
        if row is None:
            raise NotFound("outbox %s" % outbox_id)
        return self._to_record(row)

    def ready(self, *, limit: int = 100) -> List[OutboxRecord]:
        rows = self._db.all(
            "SELECT * FROM action_outbox WHERE merchant_id = :merchant_id "
            "AND status IN ('pending', 'failed') AND available_at <= :now "
            "ORDER BY created_at, id LIMIT :lim",
            self._t.scope(now=self._t.runtime.stamp(), lim=max(1, min(limit, 500))))
        return [self._to_record(row) for row in rows]

    def mark_processing(self, outbox_id: str) -> OutboxRecord:
        now = self._t.runtime.stamp()
        with self._db.tx():
            row = self.get(outbox_id)
            if row.status not in ("pending", "failed"):
                return row
            self._db.execute(
                "UPDATE action_outbox SET status = 'processing', attempts = attempts + 1, "
                "updated_at = :now WHERE merchant_id = :merchant_id AND id = :id",
                self._t.scope(id=outbox_id, now=now))
        return self.get(outbox_id)

    def succeed(self, outbox_id: str, *, provider_ref: str = "") -> OutboxRecord:
        return self._settle(outbox_id, "succeeded", provider_ref=provider_ref)

    def fail(self, outbox_id: str, error: str, *, retry_minutes: int = 5) -> OutboxRecord:
        available = self._t.runtime.clock.plus_minutes(retry_minutes)
        return self._settle(outbox_id, "failed", error=error, available_at=available)

    def _settle(self, outbox_id: str, status: str, *, provider_ref: str = "",
                error: str = "", available_at: Optional[str] = None) -> OutboxRecord:
        now = self._t.runtime.stamp()
        with self._db.tx():
            row = self.get(outbox_id)
            self._db.execute(
                "UPDATE action_outbox SET status = :status, provider_ref = :ref, "
                "last_error = :error, available_at = :available, updated_at = :now "
                "WHERE merchant_id = :merchant_id AND id = :id",
                self._t.scope(id=outbox_id, status=status, ref=provider_ref,
                              error=error, available=available_at or now, now=now))
            self._t.audit.append(
                self._t.merchant_id, "provider.intent_settled",
                {"outbox_id": outbox_id, "status": status,
                 "provider_ref": provider_ref, "error": error})
        return self.get(outbox_id)

    def _to_record(self, row: Any) -> OutboxRecord:
        return OutboxRecord(
            id=str(row["id"]), order_id=str(row["order_id"]),
            action_id=str(row["action_id"]), kind=str(row["kind"]),
            payload=canonical.loads(str(row["payload_json"])),
            idempotency_key=str(row["idempotency_key"]),
            status=str(row["status"]), attempts=int(row["attempts"]),
            provider_ref=str(row["provider_ref"]), last_error=str(row["last_error"]),
            available_at=str(row["available_at"]), created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]))


# ==========================================================================
# buyers
# ==========================================================================

class BuyerRepo:
    """Buyers, their consent, and the counters the risk features read.

    A buyer is identified by `phone_hash` within a merchant, so the same person
    ordering from two merchants is two buyers with no shared history. That is a
    tenancy requirement, not an oversight -- merchant A's RTO history is merchant
    A's data, and a cross-merchant reputation score would leak it.
    """
    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def upsert(self, *, phone: str = "", email: str = "",
               consent: Optional[Dict[Channel, bool]] = None,
               never_contact: Optional[bool] = None) -> Buyer:
        """Find or create a buyer by normalised phone. Returns the current row.

        Consent is merged rather than replaced: a second order must not silently
        revoke a permission granted at the first, and a withdrawal arrives through
        `set_consent` where it is audited.
        """
        digits = normalise_phone(phone)
        hashed = phone_hash(phone, pepper=self._t.runtime.pepper)
        if not hashed:
            # No phone at all. Still a buyer -- the order exists and its outcome
            # is a label -- but permanently uncontactable, so mark it as such
            # rather than inventing a placeholder identity.
            hashed = "nophone:" + self._t.runtime.new_id("buyer")
            never_contact = True

        now = self._t.runtime.stamp()
        with self._db.tx():
            row = self._db.one(
                "SELECT * FROM buyers "
                "WHERE merchant_id = :merchant_id AND phone_hash = :h",
                self._t.scope(h=hashed))
            if row is None:
                buyer_id = self._t.runtime.new_id("buyer")
                self._db.insert("buyers", {
                    "id": buyer_id, "merchant_id": self._t.merchant_id,
                    "phone_hash": hashed, "phone": digits.digits,
                    "email": email, "never_contact": int(bool(never_contact)),
                    "created_at": now, "updated_at": now})
            else:
                buyer_id = str(row["id"])
                updates: Dict[str, Any] = {"updated_at": now}
                # Only fill blanks. An order feed that omits the email on a repeat
                # purchase should not erase the address we already have.
                if email and not str(row["email"]):
                    updates["email"] = email
                if digits.digits and not str(row["phone"]):
                    updates["phone"] = digits.digits
                if never_contact:
                    updates["never_contact"] = 1
                self._update(buyer_id, updates)

            for channel, granted in (consent or {}).items():
                self._write_consent(buyer_id, channel, granted, now,
                                    source="order_intake")

        return self.get(buyer_id)

    def get(self, buyer_id: str) -> Buyer:
        row = self._db.one(
            "SELECT * FROM buyers WHERE merchant_id = :merchant_id AND id = :id",
            self._t.scope(id=buyer_id))
        if row is None:
            raise NotFound("no buyer %s" % buyer_id)
        consent = {
            str(c["channel"]): bool(c["granted"])
            for c in self._db.all(
                "SELECT channel, granted FROM buyer_consent "
                "WHERE merchant_id = :merchant_id AND buyer_id = :b",
                self._t.scope(b=buyer_id))}
        return Buyer(
            id=str(row["id"]), merchant_id=str(row["merchant_id"]),
            phone_hash=str(row["phone_hash"]), phone=str(row["phone"]),
            email=str(row["email"]), consent=consent,
            never_contact=bool(row["never_contact"]),
            prior_orders=int(row["prior_orders"]),
            prior_rtos=int(row["prior_rtos"]),
            prior_prepaid=int(row["prior_prepaid"]),
            prior_messages=int(row["prior_messages"]),
            prior_replies=int(row["prior_replies"]),
            last_order_at=row["last_order_at"],
            last_contacted_at=row["last_contacted_at"])

    def set_consent(self, buyer_id: str, channel: Channel, granted: bool, *,
                    source: str = "buyer") -> None:
        """Grant or withdraw consent, audited either way.

        PRD section 14 requires a withdrawal to take effect immediately and to be
        recorded. Recording a *grant* matters just as much: without it there is no
        evidence the messages already sent were permitted.
        """
        now = self._t.runtime.stamp()
        with self._db.tx():
            self._write_consent(buyer_id, channel, granted, now, source=source)
            if not granted:
                self._update(buyer_id, {"updated_at": now})
            self._t.audit.append(
                self._t.merchant_id, AuditKind.OVERRIDE_BY_MERCHANT
                if source == "merchant" else "consent.changed",
                {"buyer_id": buyer_id, "channel": channel.value,
                 "granted": int(granted), "source": source},
                actor=source)

    def set_never_contact(self, buyer_id: str, *, source: str = "buyer") -> None:
        """The global opt-out. Overrides every per-channel grant."""
        now = self._t.runtime.stamp()
        with self._db.tx():
            self._update(buyer_id, {"never_contact": 1, "updated_at": now})
            self._t.audit.append(self._t.merchant_id, "consent.never_contact",
                                 {"buyer_id": buyer_id, "source": source},
                                 actor=source)

    def record_order(self, buyer_id: str, *, placed_at: str) -> None:
        """Increment the order counter. Called once per accepted order."""
        self._db.execute(
            "UPDATE buyers SET prior_orders = prior_orders + 1, "
            "       last_order_at = :at, updated_at = :at "
            "WHERE merchant_id = :merchant_id AND id = :id",
            self._t.scope(id=buyer_id, at=placed_at))

    def record_outcome(self, buyer_id: str, outcome: OutcomeKind, *,
                       at: str) -> None:
        """Fold a terminal outcome into the buyer's history.

        Only RTO and prepaid-delivered move a counter. A cancelled order is not
        evidence about the buyer -- it is evidence about our own decision, and
        counting it as a buyer defect would let the agent manufacture its own
        justification for cancelling that buyer again.
        """
        column = {OutcomeKind.RTO: "prior_rtos",
                  OutcomeKind.PREPAID_DELIVERED: "prior_prepaid"}.get(outcome)
        if column is None:
            return
        self._db.execute(
            "UPDATE buyers SET %s = %s + 1, updated_at = :at "
            "WHERE merchant_id = :merchant_id AND id = :id" % (column, column),
            self._t.scope(id=buyer_id, at=at))

    def record_contact(self, buyer_id: str, *, at: str) -> None:
        self._db.execute(
            "UPDATE buyers SET prior_messages = prior_messages + 1, "
            "       last_contacted_at = :at, updated_at = :at "
            "WHERE merchant_id = :merchant_id AND id = :id",
            self._t.scope(id=buyer_id, at=at))

    def record_reply(self, buyer_id: str, *, at: str) -> None:
        self._db.execute(
            "UPDATE buyers SET prior_replies = prior_replies + 1, "
            "       updated_at = :at "
            "WHERE merchant_id = :merchant_id AND id = :id",
            self._t.scope(id=buyer_id, at=at))

    def messages_since(self, buyer_id: str, since: str) -> int:
        """Outbound messages since `since`. Backs the frequency cap."""
        return int(self._db.scalar(
            "SELECT COUNT(*) FROM messages "
            "WHERE merchant_id = :merchant_id AND buyer_id = :b "
            "  AND direction = 'outbound' AND status IN ('sent', 'accepted', 'delivered') "
            "  AND created_at >= :since",
            self._t.scope(b=buyer_id, since=since), default=0))

    # ------------------------------------------------------------- internals

    def _update(self, buyer_id: str, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        assignments = ", ".join("%s = :%s" % (k, k) for k in sorted(updates))
        self._db.execute(
            "UPDATE buyers SET %s "
            "WHERE merchant_id = :merchant_id AND id = :id" % assignments,
            self._t.scope(id=buyer_id, **updates))

    def _write_consent(self, buyer_id: str, channel: Channel, granted: bool,
                       now: str, *, source: str) -> None:
        self._db.execute(
            "INSERT INTO buyer_consent "
            "  (merchant_id, buyer_id, channel, granted, source, updated_at) "
            "VALUES (:merchant_id, :b, :c, :g, :src, :at) "
            "ON CONFLICT(merchant_id, buyer_id, channel) DO UPDATE SET "
            "  granted = excluded.granted, source = excluded.source, "
            "  updated_at = excluded.updated_at",
            self._t.scope(b=buyer_id, c=channel.value, g=int(bool(granted)),
                          src=source, at=now))


# ==========================================================================
# orders
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class AddressRevision:
    """One version of an order's address. Revision 1 is always as-submitted."""

    id: str
    revision: int
    source: str
    address: Address
    created_at: str


#: Who produced an address revision. Mirrors the CHECK constraint in `schema.sql`
#: so a bad source fails as a readable Python error at the call site rather than
#: as a CHECK violation from three frames down.
#:
#: The distinction that matters commercially is `buyer_confirmed` versus
#: `patched`: the first is the buyer correcting their own address, the second is
#: us normalising it without them. Only the second can damage an order that would
#: have delivered, which is what the gate's harm term prices.
ADDRESS_SOURCES = frozenset({"submitted", "buyer_confirmed", "patched",
                             "merchant_edit"})


class OrderRepo:
    """Orders, their address history, and their state machine.

    Address changes append a revision rather than overwriting, so the console can
    always show what the buyer typed next to what the agent proposed -- and a
    merchant reviewing a repair is looking at evidence, not at our word for it.
    """

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    # -------------------------------------------------------------- creating

    def create(self, *, buyer_id: str, value_paise: Paise, address: Address,
               external_id: str = "", placed_at: Optional[str] = None,
               payment_method: str = "cod", item_count: int = 1,
               category: str = "", discount_bps: int = 0,
               mode: Optional[Mode] = None,
               is_first_time_buyer: Optional[bool] = None) -> Order:
        """Accept an order at `RECEIVED`. Writes address revision 1 and audits it.

        `mode` defaults to the merchant's configured mode but may be overridden
        per order, which is how a canary runs: a slice of traffic in gateway mode
        while the rest stays advisory.
        """
        if value_paise < 0:
            raise StorageError("value_paise must not be negative")
        now = self._t.runtime.stamp()
        placed_at = placed_at or now
        order_id = self._t.runtime.new_id("order")
        effective_mode = mode or self._t.config.mode
        # Stored as NULL when absent, never as ''. Two orders that carry no
        # merchant reference are not duplicates of each other, and '' under a
        # unique index would say they were. Above the store the `Order`
        # dataclass keeps its `external_id: str = ""` contract.
        stored_external_id: Optional[str] = external_id.strip() or None

        with self._db.tx():
            if stored_external_id and self._db.exists(
                    "SELECT 1 FROM orders WHERE merchant_id = :merchant_id "
                    "AND external_id = :x",
                    self._t.scope(x=stored_external_id)):
                raise Conflict("order with external_id %r already exists"
                               % external_id)

            if is_first_time_buyer is None:
                prior = int(self._db.scalar(
                    "SELECT prior_orders FROM buyers "
                    "WHERE merchant_id = :merchant_id AND id = :b",
                    self._t.scope(b=buyer_id), default=0))
                is_first_time_buyer = prior == 0

            self._db.insert("orders", {
                "id": order_id, "merchant_id": self._t.merchant_id,
                "external_id": stored_external_id, "buyer_id": buyer_id,
                "value_paise": value_paise, "payment_method": payment_method,
                "item_count": item_count, "category": category,
                "discount_bps": discount_bps,
                "is_first_time_buyer": int(bool(is_first_time_buyer)),
                "placed_at": placed_at, "state": OrderState.RECEIVED.value,
                "mode": effective_mode.value, "created_at": now,
                "updated_at": now})
            address_id = self._append_address(order_id, address, "submitted", now)
            self._db.execute(
                "UPDATE orders SET current_address_id = :a, updated_at = :at "
                "WHERE merchant_id = :merchant_id AND id = :id",
                self._t.scope(id=order_id, a=address_id, at=now))

            BuyerRepo(self._t).record_order(buyer_id, placed_at=placed_at)

            self._t.audit.append(
                self._t.merchant_id, AuditKind.ORDER_RECEIVED,
                {"order_id": order_id, "external_id": external_id,
                 "buyer_id": buyer_id, "value_paise": value_paise,
                 "payment_method": payment_method, "mode": effective_mode.value,
                 "pincode": address.pincode, "city": address.city,
                 "item_count": item_count},
                order_id=order_id, actor="api")

        return self.get(order_id)

    # ----------------------------------------------------------------- reads

    def get(self, order_id: str) -> Order:
        row = self._db.one(
            "SELECT * FROM orders WHERE merchant_id = :merchant_id AND id = :id",
            self._t.scope(id=order_id))
        if row is None:
            raise NotFound("no order %s" % order_id)
        return self._to_order(row)

    def find(self, order_id: str) -> Optional[Order]:
        try:
            return self.get(order_id)
        except NotFound:
            return None

    def by_external_id(self, external_id: str) -> Optional[Order]:
        """Look up by the merchant's own reference. An empty reference matches nothing.

        Guarded because `external_id = NULL` never matches in SQL, so an empty
        argument would silently return `None` for a different reason than the
        caller thinks -- and a future `IS NULL` rewrite would then return an
        arbitrary reference-less order.
        """
        if not external_id.strip():
            return None
        row = self._db.one(
            "SELECT * FROM orders "
            "WHERE merchant_id = :merchant_id AND external_id = :x",
            self._t.scope(x=external_id.strip()))
        return self._to_order(row) if row is not None else None

    def address(self, order_id: str) -> Address:
        return self.current_revision(order_id).address

    def current_revision(self, order_id: str) -> AddressRevision:
        row = self._db.one(
            "SELECT a.* FROM order_addresses a "
            "JOIN orders o ON o.current_address_id = a.id "
            "WHERE a.merchant_id = :merchant_id AND o.id = :id",
            self._t.scope(id=order_id))
        if row is None:
            raise NotFound("no address for order %s" % order_id)
        return _to_revision(row)

    def address_history(self, order_id: str) -> List[AddressRevision]:
        return [_to_revision(r) for r in self._db.all(
            "SELECT * FROM order_addresses "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY revision", self._t.scope(o=order_id))]

    def in_states(self, states: Sequence[OrderState], *,
                  limit: int = 200) -> List[Order]:
        if not states:
            return []
        # Placeholders are generated from the enum, never from caller strings, so
        # this cannot be turned into an injection point.
        names = {"s%d" % i: s.value for i, s in enumerate(states)}
        clause = ", ".join(":" + k for k in sorted(names))
        return [self._to_order(r) for r in self._db.all(
            "SELECT * FROM orders WHERE merchant_id = :merchant_id "
            "AND state IN (%s) ORDER BY created_at LIMIT :lim" % clause,
            self._t.scope(lim=int(limit), **names))]

    def expired_holds(self, now: str, *, limit: int = 500) -> List[Order]:
        """Held orders whose deadline has passed. The auto-release sweep's input.

        PRD section 1: fulfilment latency is sacred. This query is the mechanism
        -- an order past `held_until` is released regardless of what the agent was
        waiting for.
        """
        return [self._to_order(r) for r in self._db.all(
            "SELECT * FROM orders WHERE merchant_id = :merchant_id "
            "  AND held_until IS NOT NULL AND held_until <= :now "
            "  AND released_at IS NULL "
            "ORDER BY held_until LIMIT :lim",
            self._t.scope(now=now, lim=int(limit)))]

    def count_by_state(self) -> Dict[str, int]:
        return {str(r["state"]): int(r["n"]) for r in self._db.all(
            "SELECT state, COUNT(*) AS n FROM orders "
            "WHERE merchant_id = :merchant_id GROUP BY state",
            self._t.scope())}

    def recent(self, *, limit: int = 100) -> List[Order]:
        """Recent tenant orders for the operator queue, including released orders."""
        return [self._to_order(row) for row in self._db.all(
            "SELECT * FROM orders WHERE merchant_id = :merchant_id "
            "ORDER BY created_at DESC, id DESC LIMIT :lim",
            self._t.scope(lim=max(1, min(int(limit), 500))))]

    # ------------------------------------------------------------ transitions

    def transition(self, order_id: str, target: OrderState, *,
                   reason: str = "", actor: str = "agent") -> Order:
        """Move an order's state, enforcing the lifecycle in `domain.TRANSITIONS`.

        Raises `IllegalTransition` rather than coercing. The audit trail is the
        product; a log that says an order went from `delivered` back to `scoring`
        because a caller asked nicely is worthless.
        """
        with self._db.tx():
            order = self.get(order_id)
            if order.state is target:
                return order          # idempotent: a repeated event is not an error
            if not can_transition(order.state, target):
                raise IllegalTransition(order.state, target)

            now = self._t.runtime.stamp()
            updates: Dict[str, Any] = {"state": target.value, "updated_at": now}
            if target is OrderState.RELEASED_FOR_DISPATCH:
                updates["released_at"] = now
                updates["held_until"] = None
            elif target in (OrderState.CANCELLED,):
                updates["held_until"] = None

            assignments = ", ".join("%s = :%s" % (k, k) for k in sorted(updates))
            self._db.execute(
                "UPDATE orders SET %s "
                "WHERE merchant_id = :merchant_id AND id = :id" % assignments,
                self._t.scope(id=order_id, **updates))
            self._db.insert("order_state_transitions", {
                "id": self._t.runtime.new_id("transition"),
                "merchant_id": self._t.merchant_id, "order_id": order_id,
                "from_state": order.state.value, "to_state": target.value,
                "reason": reason, "created_at": now})

            kind = AuditKind.ORDER_STATE_CHANGED
            if target is OrderState.RELEASED_FOR_DISPATCH:
                kind = (AuditKind.ORDER_AUTO_RELEASED
                        if actor == "timeout" else AuditKind.ORDER_RELEASED)
            self._t.audit.append(
                self._t.merchant_id, kind,
                {"order_id": order_id, "from": order.state.value,
                 "to": target.value, "reason": reason},
                order_id=order_id, actor=actor)

        return self.get(order_id)

    def hold(self, order_id: str, *, minutes: Optional[int] = None,
              reason: str = "") -> Order:
        """Set the auto-release deadline. Gateway mode only.

        The deadline is capped at `max_hold_minutes` and is written even when the
        caller asks for longer, so no code path can extend a hold past the
        merchant's stated tolerance.
        """
        limit = self._t.config.max_hold_minutes
        if minutes is None:
            wanted = limit
        else:
            wanted = int(minutes)
            if wanted < 0:
                raise StorageError("hold minutes must not be negative")
            wanted = min(wanted, limit)
        deadline = self._t.runtime.clock.plus_minutes(wanted)
        with self._db.tx():
            order = self.get(order_id)
            if order.state in TERMINAL_STATES:
                raise StorageError(
                    "cannot hold terminal order %s in state %s"
                    % (order_id, order.state.value))
            if order.state is OrderState.RELEASED_FOR_DISPATCH:
                return order
            self._db.execute(
                "UPDATE orders SET held_until = :until, updated_at = :at "
                "WHERE merchant_id = :merchant_id AND id = :id "
                "  AND released_at IS NULL",
                self._t.scope(id=order_id, until=deadline,
                              at=self._t.runtime.stamp()))
            self._t.audit.append(
                self._t.merchant_id, AuditKind.ORDER_HELD,
                {"order_id": order_id, "held_until": deadline,
                 "minutes": wanted, "cap_minutes": limit, "reason": reason},
                order_id=order_id)
        return self.get(order_id)

    def release(self, order_id: str, *, reason: str,
                actor: str = "agent") -> Order:
        """The contract event. Idempotent, and never refused for an open order."""
        order = self.get(order_id)
        if order.state is OrderState.RELEASED_FOR_DISPATCH:
            return order
        return self.transition(order_id, OrderState.RELEASED_FOR_DISPATCH,
                               reason=reason, actor=actor)

    def patch_address(self, order_id: str, address: Address, *,
                      source: str = "patched",
                      evidence: Optional[Dict[str, Any]] = None
                      ) -> AddressRevision:
        """Append a new address revision and point the order at it.

        Never overwrites. The submitted address stays at revision 1 forever, which
        is what makes an address repair reviewable rather than merely reported.
        """
        now = self._t.runtime.stamp()
        with self._db.tx():
            before = self.current_revision(order_id)
            address_id = self._append_address(order_id, address, source, now)
            self._db.execute(
                "UPDATE orders SET current_address_id = :a, updated_at = :at "
                "WHERE merchant_id = :merchant_id AND id = :id",
                self._t.scope(id=order_id, a=address_id, at=now))
            self._t.audit.append(
                self._t.merchant_id, AuditKind.ADDRESS_PATCHED,
                {"order_id": order_id, "source": source,
                 "from_revision": before.revision,
                 "to_revision": before.revision + 1,
                 "before": _address_payload(before.address),
                 "after": _address_payload(address),
                 "evidence": evidence or {}},
                order_id=order_id)
        return self.current_revision(order_id)

    # ------------------------------------------------------------- internals

    def _append_address(self, order_id: str, address: Address, source: str,
                        now: str) -> str:
        revision = 1 + int(self._db.scalar(
            "SELECT COUNT(*) FROM order_addresses "
            "WHERE merchant_id = :merchant_id AND order_id = :o",
            self._t.scope(o=order_id), default=0))
        address_id = self._t.runtime.new_id("address")
        if source not in ADDRESS_SOURCES:
            raise StorageError(
                "unknown address source %r; expected one of %s"
                % (source, ", ".join(sorted(ADDRESS_SOURCES))))
        self._db.insert("order_addresses", {
            "id": address_id, "merchant_id": self._t.merchant_id,
            "order_id": order_id, "revision": revision, "source": source,
            "raw": address.raw, "line1": address.line1, "line2": address.line2,
            "landmark": address.landmark, "locality": address.locality,
            "city": address.city, "state": address.state,
            "pincode": address.pincode, "created_at": now})
        return address_id

    def _to_order(self, row: Any) -> Order:
        return Order(
            id=str(row["id"]), merchant_id=str(row["merchant_id"]),
            buyer_id=str(row["buyer_id"]),
            value_paise=int(row["value_paise"]),
            address=self._address_for(row["current_address_id"]),
            external_id=str(row["external_id"] or ""),
            state=OrderState(str(row["state"])), mode=Mode(str(row["mode"])),
            payment_method=str(row["payment_method"]),
            item_count=int(row["item_count"]), category=str(row["category"]),
            discount_bps=int(row["discount_bps"]),
            placed_at=str(row["placed_at"]), held_until=row["held_until"],
            is_first_time_buyer=bool(row["is_first_time_buyer"]))

    def _address_for(self, address_id: Optional[str]) -> Address:
        if not address_id:
            return Address()
        row = self._db.one(
            "SELECT * FROM order_addresses "
            "WHERE merchant_id = :merchant_id AND id = :id",
            self._t.scope(id=address_id))
        return _to_revision(row).address if row is not None else Address()


def _to_revision(row: Any) -> AddressRevision:
    return AddressRevision(
        id=str(row["id"]), revision=int(row["revision"]),
        source=str(row["source"]), created_at=str(row["created_at"]),
        address=Address(
            raw=str(row["raw"]), line1=str(row["line1"]),
            line2=str(row["line2"]), landmark=str(row["landmark"]),
            locality=str(row["locality"]), city=str(row["city"]),
            state=str(row["state"]), pincode=str(row["pincode"])))


def _address_payload(address: Address) -> Dict[str, Any]:
    """Address fields for an audit payload, with the free-text lines kept.

    Unlike an export, the audit log intentionally retains the address: it is the
    record of what was changed, and redacting it would make an address repair
    unreviewable. `pii.redact` is applied at the export boundary instead.
    """
    return {"line1": address.line1, "line2": address.line2,
            "landmark": address.landmark, "locality": address.locality,
            "city": address.city, "state": address.state,
            "pincode": address.pincode}


# ==========================================================================
# the ledger
# ==========================================================================

class LedgerRepo:
    """Append-only money. Balances are sums; corrections are new rows.

    `arm` separates the backtest's counterfactual arms -- `baseline`, `rules`,
    `agent` -- from `live`. They share one table because the ROI comparison of PRD
    section 11 is then a `GROUP BY` over the same arithmetic that produced the
    live numbers, rather than a second implementation that could disagree with it.
    """

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def post(self, breakdown: LedgerBreakdown, *, order_id: Optional[str] = None,
             arm: str = "live", action_id: Optional[str] = None,
             idempotency_key: Optional[str] = None) -> List[str]:
        """Write one row per line of a breakdown. Returns the new row ids.

        The default idempotency key is `arm:order_id:outcome`, so a carrier
        reporting the same outcome twice cannot double-count a loss. A duplicate
        raises `Conflict` rather than being ignored: two different outcomes for one
        order is a real integration bug and hiding it would corrupt the ROI report.
        """
        key = idempotency_key
        if key is None and order_id is not None:
            key = "%s:%s:%s" % (arm, order_id, breakdown.outcome.value)

        now = self._t.runtime.stamp()
        ids: List[str] = []
        with self._db.tx():
            if key is not None and self._db.exists(
                    "SELECT 1 FROM ledger_entries "
                    "WHERE merchant_id = :merchant_id AND idempotency_key = :k",
                    self._t.scope(k=key)):
                raise Conflict("ledger already posted for %s" % key)

            for index, line in enumerate(breakdown.lines):
                entry_id = self._t.runtime.new_id("ledger")
                self._db.insert("ledger_entries", {
                    "id": entry_id, "merchant_id": self._t.merchant_id,
                    "order_id": order_id, "action_id": action_id,
                    "reason": line.reason.value,
                    "amount_paise": line.amount_paise, "arm": arm,
                    # Per-line suffix: the table's uniqueness is per row, and the
                    # caller-facing key is what the guard above checks.
                    "idempotency_key": None if key is None else "%s#%d" % (key, index),
                    "created_at": now})
                ids.append(entry_id)

            if key is not None:
                # A zero-amount marker carrying the caller-facing key, so the
                # duplicate check above works even for an outcome whose breakdown
                # has no lines -- `cancelled` spends nothing, and without this a
                # cancellation could be posted repeatedly.
                marker_id = self._t.runtime.new_id("ledger")
                self._db.insert("ledger_entries", {
                    "id": marker_id, "merchant_id": self._t.merchant_id,
                    "order_id": order_id, "action_id": action_id,
                    "reason": LedgerReason.OUTCOME_POSTED.value,
                    "amount_paise": 0, "arm": arm, "idempotency_key": key,
                    "created_at": now})
                ids.append(marker_id)

            self._t.audit.append(
                self._t.merchant_id, AuditKind.LEDGER_POSTED,
                {"order_id": order_id, "arm": arm,
                 "outcome": breakdown.outcome.value,
                 "net_paise": breakdown.net_paise,
                 "lines": [{"reason": l.reason.value,
                            "amount_paise": l.amount_paise}
                           for l in breakdown.lines]},
                order_id=order_id)
        return ids

    def post_action_cost(self, *, order_id: str, action_id: str,
                         cost_paise: Paise, note: str = "",
                         arm: str = "live") -> str:
        """Charge an action's cost the moment it is attempted, not when it works.

        Attempted-not-succeeded is deliberate: a message that got no reply still
        cost money, and a ledger that only records successful spending flatters
        the agent's ROI. The idempotency key is the action id, so a retry of the
        same action is charged once.
        """
        entry_id = self._t.runtime.new_id("ledger")
        with self._db.tx():
            if self._db.exists(
                    "SELECT 1 FROM ledger_entries "
                    "WHERE merchant_id = :merchant_id AND idempotency_key = :k",
                    self._t.scope(k="action:" + action_id)):
                raise Conflict("action %s already charged" % action_id)
            self._db.insert("ledger_entries", {
                "id": entry_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "action_id": action_id,
                "reason": LedgerReason.ACTION_COST.value,
                "amount_paise": -abs(cost_paise), "arm": arm,
                "idempotency_key": "action:" + action_id,
                "created_at": self._t.runtime.stamp()})
            self._t.audit.append(
                self._t.merchant_id, AuditKind.LEDGER_POSTED,
                {"order_id": order_id, "action_id": action_id, "arm": arm,
                 "reason": LedgerReason.ACTION_COST.value,
                 "amount_paise": -abs(cost_paise), "note": note},
                order_id=order_id)
        return entry_id

    def correct(self, *, order_id: Optional[str], amount_paise: Paise,
                note: str, arm: str = "live") -> str:
        """Post a compensating entry. The only way to change a balance.

        Rows are immutable at the database level, so a mistake is fixed by adding
        its negation with an explanation attached -- which leaves both the error
        and the correction visible, as an accountant would expect.
        """
        entry_id = self._t.runtime.new_id("ledger")
        with self._db.tx():
            self._db.insert("ledger_entries", {
                "id": entry_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "action_id": None,
                "reason": LedgerReason.CORRECTION.value,
                "amount_paise": amount_paise, "arm": arm,
                "idempotency_key": None,
                "created_at": self._t.runtime.stamp()})
            self._t.audit.append(
                self._t.merchant_id, AuditKind.LEDGER_POSTED,
                {"order_id": order_id, "arm": arm,
                 "reason": LedgerReason.CORRECTION.value,
                 "amount_paise": amount_paise, "note": note},
                order_id=order_id, actor="merchant")
        return entry_id

    # ----------------------------------------------------------------- reads

    def net(self, *, arm: str = "live",
            order_id: Optional[str] = None) -> Paise:
        """The balance, by summation."""
        if order_id is None:
            return int(self._db.scalar(
                "SELECT COALESCE(SUM(amount_paise), 0) FROM ledger_entries "
                "WHERE merchant_id = :merchant_id AND arm = :arm",
                self._t.scope(arm=arm), default=0))
        return int(self._db.scalar(
            "SELECT COALESCE(SUM(amount_paise), 0) FROM ledger_entries "
            "WHERE merchant_id = :merchant_id AND arm = :arm "
            "  AND order_id = :o",
            self._t.scope(arm=arm, o=order_id), default=0))

    def by_reason(self, *, arm: str = "live") -> Dict[str, Paise]:
        """Totals per reason. This is the ROI report's body."""
        return {str(r["reason"]): int(r["total"]) for r in self._db.all(
            "SELECT reason, SUM(amount_paise) AS total FROM ledger_entries "
            "WHERE merchant_id = :merchant_id AND arm = :arm "
            "GROUP BY reason ORDER BY reason", self._t.scope(arm=arm))}

    def arms(self) -> List[str]:
        return [str(r["arm"]) for r in self._db.all(
            "SELECT DISTINCT arm FROM ledger_entries "
            "WHERE merchant_id = :merchant_id ORDER BY arm", self._t.scope())]

    def entries(self, *, order_id: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._db.all(
            "SELECT reason, amount_paise, arm, action_id, created_at "
            "FROM ledger_entries "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY created_at, id", self._t.scope(o=order_id))]

    def spend_since(self, since: str, *, arm: str = "live") -> Paise:
        """Money spent on actions since `since`, positive. Backs the budget cap."""
        return -int(self._db.scalar(
            "SELECT COALESCE(SUM(amount_paise), 0) FROM ledger_entries "
            "WHERE merchant_id = :merchant_id AND arm = :arm "
            "  AND reason IN ('action_cost', 'incentive') "
            "  AND created_at >= :since",
            self._t.scope(arm=arm, since=since), default=0))


# ==========================================================================
# decisions and actions
# ==========================================================================

@dataclasses.dataclass(frozen=True)
class DecisionRecord:
    """A stored decision, complete enough to re-explain months later."""

    id: str
    order_id: str
    action: ActionKind
    expected_delta_micro: int
    threshold_ppm: int
    gate_inputs: Dict[str, Any]
    reason: str
    config_revision: int
    created_at: str


class DecisionRepo:
    """Every decision, including the ones that chose to do nothing.

    `no_action` is written as a row rather than as an absent row. Absence is
    ambiguous -- it reads the same as an order the agent never saw -- and the
    console has to be able to say "we looked at this order, here is the
    arithmetic, and the arithmetic said leave it alone". That sentence is a large
    part of what earns a merchant's trust, so it needs a row behind it.

    Append-only by trigger. A decision is a claim about what we believed at a
    moment; editing it later would make the audit trail a work of fiction.
    """

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def record(self, order_id: str, action: ActionKind, *,
               expected_delta_micro: int = 0, threshold_ppm: int = 0,
               gate_inputs: Optional[Dict[str, Any]] = None,
               reason: str = "", risk_score_id: Optional[str] = None,
               diagnosis_id: Optional[str] = None) -> DecisionRecord:
        inputs = dict(gate_inputs or {})
        # Serialised canonically, so two identical decisions produce byte-identical
        # rows and a diff between two runs of the backtest is a real difference
        # rather than a key-ordering artefact.
        payload = canonical.dumps(inputs)
        decision_id = self._t.runtime.new_id("decision")
        now = self._t.runtime.stamp()
        revision = self._t.config_revision

        with self._db.tx():
            self._db.insert("decisions", {
                "id": decision_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "risk_score_id": risk_score_id,
                "diagnosis_id": diagnosis_id, "action": action.value,
                "expected_delta_micro": int(expected_delta_micro),
                "threshold_ppm": int(threshold_ppm),
                "gate_inputs_json": payload, "reason": reason,
                "config_revision": revision, "created_at": now})
            self._t.audit.append(
                self._t.merchant_id, AuditKind.DECISION_MADE,
                {"decision_id": decision_id, "order_id": order_id,
                 "action": action.value,
                 "expected_delta_micro": int(expected_delta_micro),
                 "threshold_ppm": int(threshold_ppm), "reason": reason,
                 "config_revision": revision, "gate_inputs": inputs},
                order_id=order_id)

        return DecisionRecord(
            id=decision_id, order_id=order_id, action=action,
            expected_delta_micro=int(expected_delta_micro),
            threshold_ppm=int(threshold_ppm), gate_inputs=inputs,
            reason=reason, config_revision=revision, created_at=now)

    def suppressed(self, order_id: str, action: ActionKind, *,
                   reason: str, threshold_ppm: int = 0,
                   expected_delta_micro: int = 0,
                   gate_inputs: Optional[Dict[str, Any]] = None) -> DecisionRecord:
        """Record a decision the mode forbade acting on. Shadow mode's whole output.

        Stored with the action the gate actually chose, not with `no_action`,
        because the value of shadow mode is precisely the counterfactual: what
        would this have done, and would it have been right. Collapsing it to
        `no_action` would erase the only thing the merchant is evaluating. The
        `actions` table stays empty, so nothing downstream can mistake it for
        something that happened.
        """
        record = self.record(
            order_id, action, expected_delta_micro=expected_delta_micro,
            threshold_ppm=threshold_ppm, gate_inputs=gate_inputs, reason=reason)
        self._t.audit.append(
            self._t.merchant_id, AuditKind.DECISION_SUPPRESSED,
            {"decision_id": record.id, "order_id": order_id,
             "would_have": action.value, "reason": reason},
            order_id=order_id)
        return record

    def for_order(self, order_id: str) -> List[DecisionRecord]:
        return [self._to_record(r) for r in self._db.all(
            "SELECT * FROM decisions "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY created_at, id",
            self._t.scope(o=order_id))]

    def latest(self, order_id: str) -> Optional[DecisionRecord]:
        records = self.for_order(order_id)
        return records[-1] if records else None

    def counts_by_action(self, *, since: Optional[str] = None) -> Dict[str, int]:
        sql = ("SELECT action, COUNT(*) AS n FROM decisions "
               "WHERE merchant_id = :merchant_id")
        params = self._t.scope()
        if since:
            sql += " AND created_at >= :since"
            params["since"] = since
        return {str(r["action"]): int(r["n"])
                for r in self._db.all(sql + " GROUP BY action", params)}

    def _to_record(self, row: Any) -> DecisionRecord:
        return DecisionRecord(
            id=str(row["id"]), order_id=str(row["order_id"]),
            action=ActionKind(str(row["action"])),
            expected_delta_micro=int(row["expected_delta_micro"]),
            threshold_ppm=int(row["threshold_ppm"]),
            gate_inputs=canonical.loads(str(row["gate_inputs_json"])),
            reason=str(row["reason"]),
            config_revision=int(row["config_revision"]),
            created_at=str(row["created_at"]))


#: An action's lifecycle. `blocked` is not a failure: it records that a guardrail
#: refused the action, which is a compliance artefact the merchant may need to
#: produce later. A blocked row and a failed row must never be merged.
ACTION_STATUSES = frozenset({"pending", "succeeded", "failed", "blocked"})

# AuditKind intentionally remains a string vocabulary so old audit rows remain
# readable. This event is local to the repository until the public event vocabulary
# is versioned alongside the provider contracts.
ACTION_SETTLED_AUDIT = "action.settled"


@dataclasses.dataclass(frozen=True)
class ActionRecord:
    """One thing the agent did, or was stopped from doing."""

    id: str
    order_id: str
    kind: ActionKind
    status: str
    blocked_reason: str = ""
    cost_paise: Paise = 0
    incentive_paise: Paise = 0
    decision_id: Optional[str] = None
    created_at: str = ""

    @property
    def spent(self) -> bool:
        """Whether this action cost money. `blocked` costs nothing by definition."""
        return self.status in ("pending", "succeeded", "failed")


class ActionRepo:
    """What the agent did. The cost is charged on attempt, not on success.

    Charging on attempt is deliberate and it is the less flattering choice: a
    message that got no reply still cost money to send, and a ledger that only
    books successful spending overstates the agent's ROI by exactly the amount it
    wasted. The number this system reports about itself has to be the pessimistic
    one, or it is marketing.
    """

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def record(self, order_id: str, kind: ActionKind, *,
               status: str = "pending", decision_id: Optional[str] = None,
               cost_paise: Optional[Paise] = None, incentive_paise: Paise = 0,
               blocked_reason: str = "",
               idempotency_key: Optional[str] = None,
               arm: str = "live") -> ActionRecord:
        """Write the action row and, unless it was blocked, charge the ledger.

        One transaction: an action row without its ledger line would make the ROI
        report understate what the agent spent, which is the same class of error as
        an unposted outcome and just as invisible.
        """
        if status not in ACTION_STATUSES:
            raise StorageError("unknown action status %r; expected one of %s"
                               % (status, ", ".join(sorted(ACTION_STATUSES))))
        economics = self._t.config.economics(kind)
        cost = economics.cost_paise if cost_paise is None else int(cost_paise)
        if cost < 0 or incentive_paise < 0:
            raise StorageError("action cost and incentive must not be negative")
        if incentive_paise > self._t.config.max_incentive_paise:
            raise StorageError(
                "incentive %d exceeds this merchant's ceiling of %d"
                % (incentive_paise, self._t.config.max_incentive_paise))

        action_id = self._t.runtime.new_id("action")
        key = idempotency_key or "%s:%s:%s" % (arm, order_id, kind.value)
        now = self._t.runtime.stamp()

        with self._db.tx():
            existing = self._db.one(
                "SELECT * FROM actions "
                "WHERE merchant_id = :merchant_id AND idempotency_key = :k",
                self._t.scope(k=key))
            if existing is not None:
                # A retry, not a second action. Returning the stored row keeps a
                # duplicated webhook from charging the merchant twice for one
                # message.
                return self._to_record(existing)

            self._db.insert("actions", {
                "id": action_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "decision_id": decision_id,
                "kind": kind.value, "status": status,
                "blocked_reason": blocked_reason, "cost_paise": cost,
                "incentive_paise": int(incentive_paise),
                "idempotency_key": key, "created_at": now})

            if status == "blocked":
                self._t.audit.append(
                    self._t.merchant_id, AuditKind.ACTION_BLOCKED,
                    {"action_id": action_id, "order_id": order_id,
                     "kind": kind.value, "reason": blocked_reason},
                    order_id=order_id)
            else:
                if cost:
                    LedgerRepo(self._t).post_action_cost(
                        order_id=order_id, action_id=action_id, cost_paise=cost,
                        note=kind.value, arm=arm)
                self._t.audit.append(
                    self._t.merchant_id, AuditKind.ACTION_ATTEMPTED,
                    {"action_id": action_id, "order_id": order_id,
                     "kind": kind.value, "status": status,
                     "cost_paise": cost, "incentive_paise": int(incentive_paise),
                     "arm": arm},
                    order_id=order_id)

        return ActionRecord(
            id=action_id, order_id=order_id, kind=kind, status=status,
            blocked_reason=blocked_reason, cost_paise=cost,
            incentive_paise=int(incentive_paise), decision_id=decision_id,
            created_at=now)

    def settle(self, action_id: str, status: str, *,
               blocked_reason: str = "") -> ActionRecord:
        """Move a pending action to its final status. The cost is not refunded.

        No ledger correction on failure, on purpose: the money left the building
        when the message was sent. `LedgerRepo.correct` exists for genuine
        reversals and requires a stated reason, so an unearned refund cannot be
        slipped in as a status update.
        """
        if status not in ACTION_STATUSES:
            raise StorageError("unknown action status %r" % (status,))
        with self._db.tx():
            row = self._db.one(
                "SELECT * FROM actions "
                "WHERE merchant_id = :merchant_id AND id = :id",
                self._t.scope(id=action_id))
            if row is None:
                raise NotFound("action %s" % action_id)
            current = str(row["status"])
            if current == status and status != "pending":
                return self._to_record(row)
            if status == "pending":
                raise Conflict(
                    "action %s cannot settle to pending" % action_id)
            if current != "pending" or status == "pending":
                raise Conflict(
                    "action %s cannot settle from %s to %s"
                    % (action_id, current, status))
            self._db.execute(
                "UPDATE actions SET status = :s, blocked_reason = :r "
                "WHERE merchant_id = :merchant_id AND id = :id",
                self._t.scope(id=action_id, s=status, r=blocked_reason))
            self._t.audit.append(
                self._t.merchant_id, ACTION_SETTLED_AUDIT,
                {"action_id": action_id, "order_id": str(row["order_id"]),
                 "kind": str(row["kind"]), "from": current, "to": status,
                 "blocked_reason": blocked_reason},
                order_id=str(row["order_id"]))
            record = self._to_record(row)
        return dataclasses.replace(record, status=status,
                                   blocked_reason=blocked_reason)

    def for_order(self, order_id: str) -> List[ActionRecord]:
        return [self._to_record(r) for r in self._db.all(
            "SELECT * FROM actions "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "ORDER BY created_at, id",
            self._t.scope(o=order_id))]

    def paid_count(self, order_id: str) -> int:
        """Paid actions already taken on this order. Backs `max_paid_actions_per_order`.

        Blocked rows are excluded -- a guardrail refusing an action must not itself
        consume the order's one allowance, or a single quiet-hours block would
        silently spend the budget for an order nobody was ever contacted about.
        """
        kinds = sorted(k.value for k in PAID_ACTIONS)
        placeholders = ", ".join(":k%d" % i for i in range(len(kinds)))
        params = self._t.scope(o=order_id)
        for index, kind in enumerate(kinds):
            params["k%d" % index] = kind
        return int(self._db.scalar(
            "SELECT COUNT(*) FROM actions "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "  AND status <> 'blocked' AND kind IN (%s)" % placeholders,
            params, default=0))

    def _to_record(self, row: Any) -> ActionRecord:
        decision_id = row["decision_id"]
        return ActionRecord(
            id=str(row["id"]), order_id=str(row["order_id"]),
            kind=ActionKind(str(row["kind"])), status=str(row["status"]),
            blocked_reason=str(row["blocked_reason"]),
            cost_paise=int(row["cost_paise"]),
            incentive_paise=int(row["incentive_paise"]),
            decision_id=None if decision_id is None else str(decision_id),
            created_at=str(row["created_at"]))


# ==========================================================================
# outcomes
# ==========================================================================

class OutcomeRepo:
    """What the carrier reported. Ingested, never generated by this system.

    One method does the whole ingestion -- the outcome row, the order's terminal
    state, the ledger lines, and the buyer's counters -- inside a single
    transaction. Split across four calls, a crash between them would leave an
    order marked delivered with no money posted, or a buyer's return count
    incremented for an outcome the ledger never saw. Either one makes the ROI
    report quietly wrong, and a quietly wrong ROI report is the worst artefact
    this system could produce.
    """

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    def record(self, order_id: str, kind: OutcomeKind, *,
               reported_at: Optional[str] = None, carrier_reason: str = "",
               attempts: int = 0, source: str = "api",
               arm: str = "live") -> Outcome:
        """Ingest a terminal outcome. Idempotent for a repeat of the same outcome.

        A *different* outcome for an order that already has one raises `Conflict`.
        That is not the same as a retry: "delivered" after "rto" is a real event in
        Indian logistics -- a returned parcel redelivered on appeal -- but it is a
        reversal that costs money to unwind, so it goes through
        `LedgerRepo.correct` as a deliberate act with a note attached, rather than
        by overwriting what the ledger already believes.
        """
        now = self._t.runtime.stamp()
        reported_at = reported_at or now

        with self._db.tx():
            existing = self.find(order_id)
            if existing is not None:
                if existing.kind is kind:
                    return existing
                raise Conflict(
                    "order %s already reported as %s; %s contradicts it. A "
                    "reversal is posted as a correction, not as a second outcome."
                    % (order_id, existing.kind.value, kind.value))

            orders = OrderRepo(self._t)
            order = orders.get(order_id)
            self._walk_to_terminal(orders, order, kind)

            outcome_id = self._t.runtime.new_id("outcome")
            self._db.insert("outcomes", {
                "id": outcome_id, "merchant_id": self._t.merchant_id,
                "order_id": order_id, "kind": kind.value,
                "carrier_reason": carrier_reason, "attempts": int(attempts),
                "reported_at": reported_at, "source": source,
                "created_at": now})

            incentive = self.incentive_spent(order_id)
            breakdown = ledger_module.breakdown(
                self._t.config, order.value_paise, kind,
                incentive_paise=incentive)
            LedgerRepo(self._t).post(breakdown, order_id=order_id, arm=arm)

            BuyerRepo(self._t).record_outcome(order.buyer_id, kind, at=now)

            self._t.audit.append(
                self._t.merchant_id, AuditKind.OUTCOME_REPORTED,
                {"order_id": order_id, "kind": kind.value,
                 "carrier_reason": carrier_reason, "attempts": int(attempts),
                 "source": source, "net_paise": breakdown.net_paise,
                 "incentive_paise": incentive, "arm": arm},
                order_id=order_id, actor="carrier")

        return Outcome(order_id=order_id, merchant_id=self._t.merchant_id,
                       kind=kind, reported_at=reported_at,
                       carrier_reason=carrier_reason, attempts=int(attempts))

    def _walk_to_terminal(self, orders: "OrderRepo", order: Order,
                          kind: OutcomeKind) -> None:
        """Move the order to the state the outcome implies, one legal edge at a time.

        A real order feed skips events: an outcome can arrive for an order we last
        saw at `received`. Rather than jumping the state -- which would write an
        audit trail claiming a release that never happened -- the inferred steps are
        walked explicitly and recorded as inferred. A merchant reading the timeline
        sees exactly which events we were told about and which we filled in.
        """
        target = OUTCOME_STATE[kind]
        if order.state is target:
            return
        if order.state in TERMINAL_STATES:
            raise Conflict(
                "order %s is already %s and cannot be moved to %s"
                % (order.id, order.state.value, target.value))

        route = path_to(order.state, target)
        if route is None:
            raise Conflict("no legal path from %s to %s for order %s"
                           % (order.state.value, target.value, order.id))
        for index, step in enumerate(route):
            inferred = index < len(route) - 1
            orders.transition(
                order.id, step,
                reason=("inferred from a reported %s: this event was never sent"
                        % kind.value) if inferred
                       else "outcome reported: %s" % kind.value,
                actor="carrier")

    def incentive_spent(self, order_id: str) -> Paise:
        """What was actually paid out to convert this order, per the action log.

        Read from `actions` rather than passed in by the caller, because the
        incentive is money we spent and the record of spending it is the action
        row. An outcome payload claiming a discount that no action ever offered
        would otherwise reduce the reported margin on the strength of a carrier's
        say-so.
        """
        return int(self._db.scalar(
            "SELECT COALESCE(SUM(incentive_paise), 0) FROM actions "
            "WHERE merchant_id = :merchant_id AND order_id = :o "
            "  AND status = 'succeeded'",
            self._t.scope(o=order_id), default=0))

    def find(self, order_id: str) -> Optional[Outcome]:
        row = self._db.one(
            "SELECT * FROM outcomes "
            "WHERE merchant_id = :merchant_id AND order_id = :o",
            self._t.scope(o=order_id))
        if row is None:
            return None
        return Outcome(order_id=order_id, merchant_id=self._t.merchant_id,
                       kind=OutcomeKind(str(row["kind"])),
                       reported_at=str(row["reported_at"]),
                       carrier_reason=str(row["carrier_reason"]),
                       attempts=int(row["attempts"]))

    def counts(self, *, since: Optional[str] = None) -> Dict[str, int]:
        sql = ("SELECT kind, COUNT(*) AS n FROM outcomes "
               "WHERE merchant_id = :merchant_id")
        params = self._t.scope()
        if since:
            sql += " AND reported_at >= :since"
            params["since"] = since
        return {str(r["kind"]): int(r["n"])
                for r in self._db.all(sql + " GROUP BY kind", params)}


# ==========================================================================
# idempotency
# ==========================================================================

class IdempotencyRepo:
    """Replay protection for mutating endpoints.

    An order feed retrying a timed-out `POST /v1/orders` must not create a second
    order, and a carrier webhook redelivering an outcome must not post a second
    loss. Storing the response makes the retry return what the first call
    returned, so the client sees a success rather than a conflict it cannot act on.

    The request hash catches the other failure: a key reused with a *different*
    body, which means the client's keying is broken. That returns a conflict,
    because silently serving the first response would hide a real bug.
    """

    def __init__(self, tenant: Tenant) -> None:
        self._t = tenant
        self._db = tenant.db

    @staticmethod
    def hash_request(payload: Any) -> str:
        return canonical.sha256_hex(canonical.dumpb(payload))

    def lookup(self, key: str, endpoint: str,
               request_hash: str) -> Optional[Tuple[int, Dict[str, Any]]]:
        row = self._db.one(
            "SELECT endpoint, request_hash, response_json, status_code "
            "FROM idempotency_keys "
            "WHERE merchant_id = :merchant_id AND key = :k",
            self._t.scope(k=key))
        if row is None:
            return None
        if str(row["endpoint"]) != endpoint or str(row["request_hash"]) != request_hash:
            raise Conflict(
                "idempotency key %r was already used for a different request. "
                "Reusing a key with a changed body means the client's keying is "
                "wrong; returning the old response would hide that." % key)
        return int(row["status_code"]), canonical.loads(str(row["response_json"]))

    def remember(self, key: str, endpoint: str, request_hash: str, *,
                 status_code: int, response: Dict[str, Any]) -> None:
        self._db.execute(
            "INSERT INTO idempotency_keys "
            "  (merchant_id, key, endpoint, request_hash, response_json, "
            "   status_code, created_at) "
            "VALUES (:merchant_id, :k, :e, :h, :r, :s, :at) "
            "ON CONFLICT(merchant_id, key) DO NOTHING",
            self._t.scope(k=key, e=endpoint, h=request_hash,
                          r=canonical.dumps(response), s=int(status_code),
                          at=self._t.runtime.stamp()))
