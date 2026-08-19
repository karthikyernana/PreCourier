"""Phone numbers and other personal data: normalise, hash, mask.

One module, because these three operations must agree everywhere and each has a
way of going quietly wrong if reimplemented per call site.

*   **Normalise before hashing.** `+91 98765 43210`, `098765 43210` and
    `9876543210` are one buyer. Hash them as typed and they become three, the
    repeat-returner feature never fires, and the model learns nothing from the
    signal PRD section 4 leans on hardest.

*   **Hash for joins, keep plaintext only where dispatch requires it.** PRD
    section 14: analytics and exports use `phone_hash`; the plaintext exists
    solely because a verification message has to reach a handset.

*   **Mask for display.** The console shows `98••••3210`. An ops user needs to
    tell two buyers apart; they do not need a bulk-exportable contact list.

Indian mobile numbers are the case this handles properly: ten digits, first digit
6-9, optionally prefixed +91/91/0. Anything else is normalised on a best-effort
basis and flagged implausible rather than rejected -- a merchant's real data
contains landlines and typos, and dropping those orders would lose their labels.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Optional

#: Country code we strip. Single-country by design: this product is COD in India,
#: and pretending otherwise would mean a phone library and a locale config for a
#: market we do not serve.
IN_COUNTRY_CODE = "91"

_DIGITS = re.compile(r"\D+")
_PLAUSIBLE_MOBILE = re.compile(r"^[6-9]\d{9}$")


@dataclasses.dataclass(frozen=True)
class Phone:
    """A normalised phone number and what we could tell about it.

    `plausible` feeds the `unreachable_contact` detector of PRD section 4. It is
    a structural judgement -- ten digits starting 6-9 -- not a claim that the
    number is in service. Only a real call or message can establish that, which
    is precisely what `contact_verify` buys.
    """

    digits: str
    plausible: bool
    original: str

    def __bool__(self) -> bool:
        return bool(self.digits)

    @property
    def e164(self) -> str:
        """`+919876543210`, the form a messaging provider wants."""
        if not self.digits:
            return ""
        if len(self.digits) == 10:
            return "+%s%s" % (IN_COUNTRY_CODE, self.digits)
        return "+" + self.digits

    def masked(self) -> str:
        return mask_phone(self.digits)


def normalise_phone(raw: Optional[str]) -> Phone:
    """Strip formatting and the Indian country code down to ten digits.

    Deliberately does not reject implausible input. A merchant's order feed will
    contain landlines, nine-digit typos and the occasional `0000000000`; refusing
    those orders would discard their RTO labels, and an implausible number is
    itself the strongest single predictor of an unreachable buyer.
    """
    original = (raw or "").strip()
    digits = _DIGITS.sub("", original)

    # +91 / 91 / 0 prefixes, longest first so "910" is not mistaken for "91".
    if len(digits) == 12 and digits.startswith(IN_COUNTRY_CODE):
        digits = digits[2:]
    elif len(digits) == 13 and digits.startswith("0" + IN_COUNTRY_CODE):
        digits = digits[3:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]

    return Phone(digits=digits,
                 plausible=bool(_PLAUSIBLE_MOBILE.match(digits)),
                 original=original)


def mask_phone(digits: str) -> str:
    """`9876543210` -> `98••••3210`. What the console renders.

    Keeps the first two and last four: enough for an ops user to match a number
    against a courier's call log, not enough to dial.
    """
    cleaned = _DIGITS.sub("", digits or "")
    if not cleaned:
        return "(no number)"
    if len(cleaned) <= 4:
        return "•" * len(cleaned)
    head, tail = cleaned[:2], cleaned[-4:]
    return "%s%s%s" % (head, "•" * max(1, len(cleaned) - 6), tail)


def mask_email(email: str) -> str:
    """`arjun.mehta@example.com` -> `a••••a@example.com`."""
    local, _, domain = (email or "").partition("@")
    if not domain:
        return "(no email)" if not local else "•" * len(local)
    if len(local) <= 2:
        return "%s@%s" % ("•" * len(local), domain)
    return "%s%s%s@%s" % (local[0], "•" * (len(local) - 2), local[-1], domain)


def phone_hash(raw: Optional[str], *, pepper: str = "") -> str:
    """Stable identity for a buyer, computed from the normalised digits.

    Normalisation happens here rather than at the call site so a caller cannot
    accidentally hash the unnormalised string and split one buyer into several.
    """
    from .runtime import hash_secret
    normalised = normalise_phone(raw)
    if not normalised.digits:
        return ""
    return hash_secret("phone:" + normalised.digits, pepper=pepper)


#: Keys never written to an export, a report, or a log line. `reporting` filters
#: against this set, and a test asserts a generated report contains none of them.
PII_KEYS = frozenset({
    "phone", "phone_number", "mobile", "email", "buyer_name", "name",
    "line1", "line2", "landmark", "raw", "address_text", "recipient",
})


def redact(payload: dict) -> dict:
    """Recursively drop PII keys. Used on anything leaving the tenant boundary.

    Key-name based, which is a blunt instrument -- it cannot catch a phone number
    embedded in a free-text field. That gap is closed separately: message bodies
    are excluded from exports wholesale rather than filtered.
    """
    out = {}
    for key, value in payload.items():
        if key in PII_KEYS:
            continue
        if isinstance(value, dict):
            out[key] = redact(value)
        elif isinstance(value, list):
            out[key] = [redact(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out
