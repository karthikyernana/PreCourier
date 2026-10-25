"""Provider contracts at the production boundary.

These interfaces define what a real store, carrier, messaging, or payment provider
must implement. They do not instantiate vendors or claim live integration.
"""

from __future__ import annotations

import dataclasses
import os
import httpx
from typing import Any, Dict, Mapping, Protocol

from .domain import Address, Channel, Order
from .runtime import parse_ts
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
    def create(self, order: Order, *, amount_paise: int, idempotency_key: str,
               expire_by: int | str = 0,
               notify: Mapping[str, bool] | None = None,
               callback_url: str = "", accept_partial: bool = False
               ) -> Dict[str, Any]: ...


class MessagingProvider(Protocol):
    def send(self, order: Order, *, body: str, idempotency_key: str) -> ProviderMessage: ...


class FakePaymentLinkProvider:
    provider_name = "fake"

    def create(self, order: Order, *, amount_paise: int, idempotency_key: str,
               expire_by: int | str = 0,
               notify: Mapping[str, bool] | None = None,
               callback_url: str = "", accept_partial: bool = False
               ) -> Dict[str, Any]:
        return {
            "id": "fake_" + idempotency_key,
            "short_url": "https://pay.invalid/" + idempotency_key,
            "amount": amount_paise,
            "status": "created",
            "expire_by": expire_by,
            "notify": dict(notify or {}),
            "callback_url": callback_url,
            "accept_partial": accept_partial,
        }


class FakeMessagingProvider:
    def send(self, order: Order, *, body: str, idempotency_key: str) -> ProviderMessage:
        return ProviderMessage(True, "fake", idempotency_key, body)


class RazorpayPaymentLinkProvider:
    """Razorpay test-mode adapter. Transport is injectable and never runs in tests."""
    provider_name = "razorpay"

    def __init__(self, transport: Any = None, *, key_id: str | None = None,
                 key_secret: str | None = None, callback_url: str | None = None):
        self.transport = transport or httpx.Client()
        self.key_id = key_id or os.environ.get("RAZORPAY_KEY_ID", "")
        self.key_secret = key_secret or os.environ.get("RAZORPAY_KEY_SECRET", "")
        self.callback_url = callback_url or os.environ.get(
            "RAZORPAY_CALLBACK_URL",
            "https://precourier.invalid/v1/webhooks/razorpay")

    def create(self, order: Order, *, amount_paise: int, idempotency_key: str,
               expire_by: int | str = 0,
               notify: Mapping[str, bool] | None = None,
               callback_url: str = "", accept_partial: bool = False
               ) -> Dict[str, Any]:
        if not self.key_id or not self.key_secret:
            raise ProviderError("Razorpay credentials are not configured")
        response = self._post(
            "https://api.razorpay.com/v1/payment_links",
            auth=(self.key_id, self.key_secret),
            headers={"X-Idempotency-Key": idempotency_key},
            json={
                # Razorpay's API is integer paise; the service never gives this
                # adapter a rupee float to round.
                "amount": int(amount_paise),
                "currency": "INR",
                "accept_partial": bool(accept_partial),
                "description": "PreCourier order " + order.id,
                "reference_id": order.id,
                "expire_by": _unix_seconds(expire_by),
                "notify": dict(notify or {"sms": True, "email": True}),
                "customer_notify": True,
                "callback_url": callback_url or self.callback_url,
                "callback_method": "get",
            })
        response.raise_for_status()
        return response.json()

    def _post(self, url: str, **kwargs: Any) -> Any:
        """Accept either an httpx.Client or a raw MockTransport seam."""
        if hasattr(self.transport, "post"):
            return self.transport.post(url, **kwargs)
        with httpx.Client(transport=self.transport) as client:
            return client.post(url, **kwargs)


def _unix_seconds(value: int | str) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return int(parse_ts(text).timestamp())


def default_payment_link_provider() -> PaymentLinkProvider:
    """Select the fake provider unless both Razorpay credentials are present."""
    if os.environ.get("RAZORPAY_KEY_ID") and os.environ.get("RAZORPAY_KEY_SECRET"):
        return RazorpayPaymentLinkProvider()
    return FakePaymentLinkProvider()


__all__ = ["CarrierAdapter", "CarrierAddressResult", "MessagingAdapter", "PaymentLinkProvider", "MessagingProvider", "FakePaymentLinkProvider", "FakeMessagingProvider", "RazorpayPaymentLinkProvider", "default_payment_link_provider",
           "PaymentAdapter", "ProviderError", "ProviderMessage", "ProviderUnknown",
           "ServiceabilityResult", "StoreAdapter"]
