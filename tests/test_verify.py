"""Deterministic address verification boundaries for Indian PIN-based delivery."""

from types import SimpleNamespace

from precourier.domain import Address
from precourier.verify import verify_address_patch


class _Reference:
    def get(self, pincode):
        if pincode != "560034":
            return None
        return SimpleNamespace(
            primary_place="Bengaluru", district="Bengaluru Urban",
            places=("Koramangala",))


ORIGINAL = Address(line1="Flat 402, Residency", locality="Koramangala",
                   city="Bengaluru", state="Karnataka", pincode="560034")


def test_verifier_rejects_the_named_pin_and_locality_failures():
    reference = _Reference()
    assert verify_address_patch(
        ORIGINAL, Address(line1="Flat 402", city="Bengaluru", pincode="000000"),
        reference=reference).code == "pincode_unknown"
    assert verify_address_patch(
        ORIGINAL, Address(line1="Flat 402", city="Mumbai", pincode="560034"),
        reference=reference).code == "pincode_city_mismatch"
    assert verify_address_patch(
        ORIGINAL, Address(line1="Flat 402", city="Bengaluru", pincode="560034"),
        reference=reference).code == "locality_dropped"


def test_verifier_rejects_aggressive_or_carrier_rejected_patches():
    aggressive = Address(
        raw=("one two three four five six seven eight nine ten eleven twelve "
             "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
             "twenty twentyone twentytwo twentythree twentyfour twentyfive "
             "twentysix twent..."),
        locality="Koramangala", city="Bengaluru", pincode="560034")
    assert verify_address_patch(ORIGINAL, aggressive,
                                reference=_Reference()).code == "rewrite_too_aggressive"
    candidate = Address(line1="Flat 402, Residency", locality="Koramangala",
                        city="Bengaluru", pincode="560034")
    assert verify_address_patch(ORIGINAL, candidate, reference=_Reference(),
                                carrier_valid=False).code == "carrier_rejected"


def test_verifier_accepts_an_additive_patch_that_preserves_locality():
    candidate = Address(line1="Flat 402, Residency", line2="12th Main Road",
                        landmark="Opposite Sony World", locality="Koramangala",
                        city="Bengaluru", state="Karnataka", pincode="560034")
    assert verify_address_patch(ORIGINAL, candidate,
                                reference=_Reference()).accepted is True
