"""Deterministic validation for address patches proposed by a buyer or LLM."""

from __future__ import annotations

import dataclasses
import re
from typing import Any, Optional, Tuple

from .domain import Address


@dataclasses.dataclass(frozen=True)
class Verification:
    accepted: bool
    code: str = ""
    detail: str = ""


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"[\w]+", value)
            if len(token) > 1}


def _same_city(left: str, right: str) -> bool:
    aliases = {"bengaluru": "bangalore", "bombay": "mumbai",
               "madras": "chennai", "calcutta": "kolkata"}
    def normal(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "", value.casefold())
        for old, new in aliases.items():
            cleaned = cleaned.replace(old, new)
        return cleaned
    return bool(normal(left)) and normal(left) == normal(right)


def verify_address_patch(original: Address, candidate: Address, *,
                         reference: Optional[Any] = None,
                         carrier_valid: Optional[bool] = None) -> Verification:
    """Verify identity-preserving, additive Indian address changes.

    The original address remains the source of truth on every rejection. This
    verifier checks structural facts only; it does not geocode or claim carrier
    serviceability without an injected carrier verdict.
    """
    pincode = candidate.pincode.strip()
    if reference is not None:
        row = reference.get(pincode) if pincode else None
        if row is None:
            return Verification(False, "pincode_unknown",
                                "the candidate PIN is not in the reference")
        places = tuple(getattr(row, "places", ())) + (
            str(getattr(row, "primary_place", "")),
            str(getattr(row, "district", "")))
        if candidate.city and places and not any(
                _same_city(candidate.city, place) for place in places if place):
            return Verification(False, "pincode_city_mismatch",
                                "the candidate city does not match the PIN")
    original_locality = _tokens(original.locality)
    candidate_text = _tokens(candidate.as_text())
    if original_locality and not original_locality.intersection(candidate_text):
        return Verification(False, "locality_dropped",
                            "the candidate removed the original locality")
    original_tokens = _tokens(original.as_text())
    candidate_tokens = _tokens(candidate.as_text())
    if original_tokens and len(candidate_tokens) > max(3, len(original_tokens) * 3):
        return Verification(False, "rewrite_too_aggressive",
                            "the candidate rewrites too much of the submitted address")
    if carrier_valid is False:
        return Verification(False, "carrier_rejected",
                            "the carrier rejected the candidate address")
    return Verification(True)


__all__ = ["Verification", "verify_address_patch"]
