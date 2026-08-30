"""Provider contracts at the production boundary.

These interfaces define what a real store, carrier, messaging, or payment provider
must implement. They do not instantiate vendors or claim live integration.
"""

from __future__ import annotations

import dataclasses
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


__all__ = ["CarrierAdapter", "CarrierAddressResult", "MessagingAdapter",
           "PaymentAdapter", "ProviderError", "ProviderMessage", "ProviderUnknown",
           "ServiceabilityResult", "StoreAdapter"]
