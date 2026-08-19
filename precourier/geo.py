"""India postal-code geography, read from the real GeoNames-derived table.

The one module that opens `data/pincodes.csv.gz`. It exists because three separate
consumers need the same answers and must not disagree about them: the feature
builder asks whether a pincode is real and how far it is from the merchant's
origin, the verification layer asks whether the pincode contradicts the city the
buyer typed, and the simulated carrier asks whether an address is serviceable.
Three copies of that lookup would drift, and the drift would show up as a model
scoring geography no verifier ever checked.

Two properties matter more than speed.

**It is real data.** 19,238 pincodes with real coordinates, real districts and a
derived density measure -- see `data/SOURCES.md` for the licence and the retrieval
date. The synthetic world places buyers at genuine pincodes, so the distance
feature measures a distance that exists and a pincode that a courier would
recognise. A generator that invented six-digit numbers would produce a model whose
geography feature was noise dressed as signal.

**It is stdlib-only and lazy.** PRD section 9 requires the backtest to run with
nothing installed, and `csv` plus `gzip` plus `math` is all this needs. The table
loads on first use rather than at import, because most processes that import
`precourier` never ask a geography question and 19,238 rows is not free.
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import gzip
import math
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "data", "pincodes.csv.gz")

#: The tier vocabulary, densest first. Read off `tools/build_pincode_reference.py`'s
#: `TIER_CUTS` -- the builder cuts measured 10km pincode density at the 95th, 75th
#: and 40th percentiles, so these four labels are exhaustive and every one is
#: populated. Ordered so that "at least tier_1" is expressible and so the console
#: renders them in a stable sequence.
TIERS: Tuple[str, ...] = ("metro", "tier_1", "tier_2", "tier_3")

#: Mean earth radius, kilometres. Distances here are used for a *band* feature
#: (0-50km, 50-200km, ...), so the ellipsoid correction a geodesic library would
#: add is far below the width of the bands it feeds.
EARTH_RADIUS_KM = 6371.0088


class GeoError(RuntimeError):
    """The reference table is missing or unreadable."""


@dataclasses.dataclass(frozen=True)
class Pincode:
    """One row of the reference table.

    `places` is every place name the upstream data lists under this pincode. It is
    the basis of the `locality_in_reference` check: a locality the buyer typed that
    appears nowhere in this list is evidence of an address defect, and a locality
    that does appear is evidence against one. Kept as a tuple so a caller cannot
    mutate the shared row.
    """

    pincode: str
    state: str
    district: str
    primary_place: str
    lat: float
    lon: float
    tier: str
    density_10km: int
    places: Tuple[str, ...] = ()

    @property
    def is_metro(self) -> bool:
        return self.tier == "metro"

    def mentions(self, locality: str) -> bool:
        """Whether `locality` matches any place name under this pincode.

        Substring in both directions, case-folded. "Koramangala" should match
        "Koramangala 4th Block" and the reverse, because a buyer writes the short
        form and the postal table holds the long one. Both directions are needed:
        neither string is reliably the longer.
        """
        needle = locality.strip().casefold()
        if not needle:
            return False
        for place in self.places:
            hay = place.casefold()
            if needle in hay or hay in needle:
                return True
        return False


class Reference:
    """The loaded table. One instance is shared; construct your own only in tests.

    Thread-safe on load. The API layer serves sync endpoints from a threadpool, so
    two requests can race the first lookup, and a half-populated dict would answer
    "unknown pincode" for a real one -- which the defect detector would then read as
    evidence of a bad address. A lock around a one-off load is cheap insurance
    against an intermittently wrong feature.
    """

    def __init__(self, path: str = DATA_PATH) -> None:
        self._path = path
        self._rows: Dict[str, Pincode] = {}
        self._sorted: List[str] = []
        self._lock = threading.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:                 # another thread won the race
                return
            if not os.path.exists(self._path):
                raise GeoError(
                    "pincode reference table missing at %s. Rebuild it with "
                    "`python3 tools/build_pincode_reference.py`." % self._path)
            rows: Dict[str, Pincode] = {}
            with gzip.open(self._path, "rt", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    code = (row.get("pincode") or "").strip()
                    if not code:
                        continue
                    rows[code] = Pincode(
                        pincode=code,
                        state=(row.get("state") or "").strip(),
                        district=(row.get("district") or "").strip(),
                        primary_place=(row.get("primary_place") or "").strip(),
                        lat=float(row.get("lat") or 0.0),
                        lon=float(row.get("lon") or 0.0),
                        tier=(row.get("tier") or "tier_3").strip(),
                        density_10km=int(row.get("density_10km") or 0),
                        places=tuple(
                            p for p in (row.get("places") or "").split("|") if p))
            if not rows:
                raise GeoError("pincode reference table at %s is empty"
                               % self._path)
            self._rows = rows
            # Sorted once, so sampling a world is reproducible: iterating a dict
            # is insertion-ordered in CPython but the insertion order here is the
            # file's, and a rebuilt file could reorder. A sorted list makes the
            # sampling order a property of the *data* rather than of the build.
            self._sorted = sorted(rows)
            self._loaded = True

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        self._load()
        return len(self._rows)

    def __contains__(self, pincode: object) -> bool:
        self._load()
        return str(pincode).strip() in self._rows

    def get(self, pincode: str) -> Optional[Pincode]:
        """The row, or `None` for a pincode that is not in the reference.

        `None` is a real answer, not an error: a buyer typing a pincode that does
        not exist is one of the commonest address defects and the caller wants to
        record it as evidence rather than to handle an exception.
        """
        self._load()
        return self._rows.get(str(pincode).strip())

    def codes(self) -> Sequence[str]:
        """Every pincode, ascending. The sampling population for `worldgen`."""
        self._load()
        return self._sorted

    def by_tier(self, tier: str) -> Sequence[str]:
        self._load()
        return [c for c in self._sorted if self._rows[c].tier == tier]

    def nearest(self, pincode: str) -> Optional[str]:
        """The numerically adjacent pincode that exists.

        Used to build a *plausible* wrong pincode for the synthetic world: a
        transposed digit that lands on a real code in a different district is a far
        better defect than a random six digits, because it is the mistake buyers
        actually make and it is the one a pincode-validity check cannot catch.
        """
        self._load()
        code = str(pincode).strip()
        idx = bisect.bisect_left(self._sorted, code)
        candidates = []
        if idx < len(self._sorted):
            candidates.append(self._sorted[idx])
        if idx > 0:
            candidates.append(self._sorted[idx - 1])
        for candidate in candidates:
            if candidate != code:
                return candidate
        return None


#: The shared instance. Module-level rather than a singleton function because the
#: table is immutable once loaded and there is exactly one of it per process.
REFERENCE = Reference()


# ==========================================================================
# distance
# ==========================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def distance_km(origin: str, destination: str,
                reference: Optional[Reference] = None) -> Optional[float]:
    """Distance between two pincodes, or `None` if either is unknown.

    `None` rather than a large sentinel. A sentinel would flow into the distance
    band as "very far", and an unknown pincode is a *different* fact from a distant
    one -- one is an address defect, the other is ordinary long-haul shipping. The
    feature builder encodes them separately.
    """
    ref = reference or REFERENCE
    a, b = ref.get(origin), ref.get(destination)
    if a is None or b is None:
        return None
    return haversine_km(a.lat, a.lon, b.lat, b.lon)


#: Upper bounds in km. The last band is open-ended.
DISTANCE_BANDS: Tuple[int, ...] = (50, 200, 800, 1800)


def distance_band(km: Optional[float]) -> str:
    """A coarse label for the distance, or `unknown`.

    Bands rather than raw kilometres because the relationship with RTO is not
    linear -- intracity, intrastate, and cross-country are three different
    operational realities with different handover counts -- and because a banded
    feature is one a merchant can read in the console.
    """
    if km is None:
        return "unknown"
    for i, bound in enumerate(DISTANCE_BANDS):
        if km < bound:
            return "band_%d" % i
    return "band_%d" % len(DISTANCE_BANDS)


__all__ = ["DATA_PATH", "DISTANCE_BANDS", "REFERENCE", "TIERS", "GeoError",
           "Pincode", "Reference", "distance_band", "distance_km",
           "haversine_km"]
