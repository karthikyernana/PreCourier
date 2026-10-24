"""Provider contracts at the production boundary.

These interfaces define what a real store, carrier, messaging, or payment provider
must implement. They do not instantiate vendors or claim live integration.
"""

from __future__ import annotations

import dataclasses
import os
import httpx
from typing import Any, Dict, Protocol

from .domain import Address, Channel, Order
from .states import Verdict


class ProviderError(RuntimeError):
    """A provider call failed or returned an unusable response."""


class ProviderUnknown(ProviderError):
    """The provider may have accepted the request but its result is unknown."""


@dataclasses.dataclass(frozen=True)
class ServiceabilityResult:
    serviceable: bool
    provider: str
    reference: str = ""
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class CarrierAddressResult:
    accepted: bool
    provider: str
    reference: str = ""
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class ProviderMessage:
    accepted: bool
    provider: str
    provider_ref: str = ""
    detail: str = ""


class StoreAdapter(Protocol):
    def apply_address_patch(self, order: Order, address: Address) -> None: ...
    def release_for_dispatch(self, order: Order) -> None: ...
    def cancel_before_dispatch(self, order: Order, reason: str) -> None: ...


class CarrierAdapter(Protocol):
    def check_serviceability(self, address: Address, *, payment_method: str
                             ) -> ServiceabilityResult: ...
    def validate_address(self, address: Address) -> CarrierAddressResult: ...
    def hold_label(self, order: Order) -> None: ...
    def release_label(self, order: Order) -> None: ...


class MessagingAdapter(Protocol):
    channel: Channel
    def send(self, order: Order, verdict: Verdict, *, body: str,
             idempotency_key: str) -> ProviderMessage: ...


class PaymentAdapter(Protocol):
    def create_payment_link(self, order: Order, *, amount_paise: int,
                            idempotency_key: str) -> Dict[str, Any]: ...
    def verify_callback(self, body: bytes, signature: str) -> Dict[str, Any]: ...


class PaymentLinkProvider(Protocol):
    def create(self, order: Order, *, amount_paise: int, idempotency_key: str) -> Dict[str, Any]: ...


class MessagingProvider(Protocol):
    def send(self, order: Order, *, body: str, idempotency_key: str) -> ProviderMessage: ...


class FakePaymentLinkProvider:
    def create(self, order: Order, *, amount_paise: int, idempotency_key: str) -> Dict[str, Any]:
        return {"id": "fake_" + idempotency_key, "short_url": "https://pay.invalid/" + idempotency_key,
                "amount": amount_paise, "status": "created"}


class FakeMessagingProvider:
    def send(self, order: Order, *, body: str, idempotency_key: str) -> ProviderMessage:
        return ProviderMessage(True, "fake", idempotency_key, body)


class RazorpayPaymentLinkProvider:
    """Razorpay test-mode adapter. Transport is injectable and never runs in tests."""
    def __init__(self, transport: Any = None, *, key_id: str | None = None, key_secret: str | None = None):
        self.transport = transport or httpx.Client()
        self.key_id = key_id or os.environ.get("RAZORPAY_KEY_ID", "")
        self.key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET", "")

    def create(self, order: Order, *, amount_paise: int, idempotency_key: str) -> Dict[str, Any]:
        if not self.key_id or not self.key_secret:
            raise ProviderError("Razorpay credentials are not configured")
        response = self.transport.post("https://api.razorpay.com/v1/payment_links",
            auth=(self.key_id, self.key_secret), headers={"X-Idempotency-Key": idempotency_key},
            json={"amount": amount_paise, "currency": "INR", "accept_partial": False,
                  "description": "PreCourier order " + order.id, "reference_id": order.id})
        response.raise_for_status()
        return response.json()


__all__ = ["CarrierAdapter", "CarrierAddressResult", "MessagingAdapter", "PaymentLinkProvider", "MessagingProvider", "FakePaymentLinkProvider", "FakeMessagingProvider", "RazorpayPaymentLinkProvider",
           "PaymentAdapter", "ProviderError", "ProviderMessage", "ProviderUnknown",
           "ServiceabilityResult", "StoreAdapter"]
