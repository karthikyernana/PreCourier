#!/usr/bin/env python3
"""Build the pincode reference table from the GeoNames India postal dataset.

Run:
    python3 tools/build_pincode_reference.py

Source
------
GeoNames postal-code export for India, https://download.geonames.org/export/zip/IN.zip
Licensed CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Attribution is
recorded in `precourier/data/SOURCES.md` and rendered in the console footer.

Why a build script rather than a committed blob
-----------------------------------------------
The reference table is the one place this system asserts a fact about the real
world -- that a pincode exists and belongs to a particular district. A committed
CSV with no derivation is an unfalsifiable claim. This script makes the
derivation reproducible: re-run it against a fresh download and the output is
identical modulo GeoNames' own updates.

Output
------
`precourier/data/pincodes.csv.gz`, one row per pincode:

    pincode, state, district, primary_place, lat, lon,
    tier, density_10km, n_places, places

`places` is a pipe-joined list of every place name GeoNames records for the
pincode. It backs two things the agent needs: the `locality_in_reference`
feature, and the `locality_dropped` verification check that stops an LLM from
rewriting a buyer's locality out of their address.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import statistics
import sys
from typing import Dict, List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO, "var", "raw", "IN.txt")
OUT_DIR = os.path.join(REPO, "precourier", "data")
OUT = os.path.join(OUT_DIR, "pincodes.csv.gz")


# --------------------------------------------------------------------------
# City tier, from measured pincode density.
#
# An earlier version of this script classified tier by matching district names
# against an editorial list. That was wrong twice over: Connaught Place
# (110001) looked rural because GeoNames records 21 named sub-localities inside
# one dense urban pincode, and Gurgaon fell through because the list spelled it
# "Gurugram". Name matching cannot be made complete and its failures are silent.
#
# Tier is therefore derived from a quantity actually present in the data: how
# many distinct pincode centroids lie within 10 km. Urban cores pack dozens of
# pincodes into a few square kilometres; a rural pincode may be the only one for
# 30 km. The metro name list survives only as a floor -- a Delhi pincode is
# never labelled rural -- and `density_10km` ships as a numeric feature in its
# own right, because a model can use the count better than the bucket.
# --------------------------------------------------------------------------
METRO = {
    "mumbai", "mumbai suburban", "delhi", "new delhi", "central delhi",
    "north delhi", "south delhi", "east delhi", "west delhi", "north east delhi",
    "north west delhi", "south west delhi", "south east delhi", "shahdara",
    "bengaluru urban", "bangalore urban", "bangalore", "hyderabad", "chennai",
    "kolkata", "pune", "ahmedabad", "ahmadabad",
}

#: Radius for the density count, in kilometres.
DENSITY_RADIUS_KM = 10.0

#: Grid cell for the spatial prefilter, in degrees. A 3x3 neighbourhood of
#: 0.1-degree cells spans at least 14 km on both axes at Indian latitudes,
#: which safely contains a 10 km radius.
GRID = 0.1

#: Density percentile cut points, declared so a reviewer can argue with the cuts
#: rather than with a black box.
TIER_CUTS = (("metro", 95), ("tier_1", 75), ("tier_2", 40))

EARTH_RADIUS_KM = 6371.0

UPSTREAM_URL = "https://download.geonames.org/export/zip/IN.zip"


def ensure_source() -> bool:
    """Download and unzip the upstream export if it is not already present.

    Makes `python3 tools/build_pincode_reference.py` work from a fresh clone
    with nothing installed -- `var/raw/` is gitignored, so the derived table can
    always be regenerated rather than trusted.
    """
    if os.path.exists(SOURCE):
        return True
    import urllib.request
    import zipfile

    raw_dir = os.path.dirname(SOURCE)
    os.makedirs(raw_dir, exist_ok=True)
    archive = os.path.join(raw_dir, "IN.zip")
    print("downloading %s" % UPSTREAM_URL, file=sys.stderr)
    try:
        urllib.request.urlretrieve(UPSTREAM_URL, archive)
    except OSError as exc:
        print("download failed: %s" % exc, file=sys.stderr)
        return False
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    return os.path.exists(SOURCE)


def normalise(name: str) -> str:
    return " ".join(name.strip().lower().split())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def compute_density(centroids: Dict[str, Tuple[float, float]]) -> Dict[str, int]:
    """Pincodes within DENSITY_RADIUS_KM of each pincode, itself excluded.

    Grid-bucketed, so this is linear in practice rather than the 370 million
    pair comparisons a naive scan over 19k pincodes would need.
    """
    grid: Dict[Tuple[int, int], List[str]] = {}
    for pincode, (lat, lon) in centroids.items():
        grid.setdefault((int(lat / GRID), int(lon / GRID)), []).append(pincode)

    density: Dict[str, int] = {}
    for pincode, (lat, lon) in centroids.items():
        cell = (int(lat / GRID), int(lon / GRID))
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((cell[0] + dx, cell[1] + dy), ()):
                    if other == pincode:
                        continue
                    olat, olon = centroids[other]
                    if haversine_km(lat, lon, olat, olon) <= DENSITY_RADIUS_KM:
                        count += 1
        density[pincode] = count
    return density


def tier_from_density(density: int, thresholds: Dict[str, int],
                     district: str) -> str:
    if normalise(district) in METRO:
        # A floor, not an override: a metro pincode may legitimately be sparse
        # at the city edge, but it is never rural.
        return "metro" if density >= thresholds["metro"] else "tier_1"
    for label, _pct in TIER_CUTS:
        if density >= thresholds[label]:
            return label
    return "tier_3"


def load(path: str) -> Dict[str, List[Tuple[str, str, str, float, float]]]:
    """pincode -> [(place, state, district, lat, lon), ...]"""
    by_pin: Dict[str, List[Tuple[str, str, str, float, float]]] = {}
    skipped = 0
    with open(path, encoding="utf-8") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            # country, postal, place, admin1, code1, admin2, code2,
            # admin3, code3, lat, lon, accuracy
            if len(row) < 12:
                skipped += 1
                continue
            pincode = row[1].strip()
            if len(pincode) != 6 or not pincode.isdigit():
                skipped += 1
                continue
            try:
                lat, lon = float(row[9]), float(row[10])
            except ValueError:
                skipped += 1
                continue
            by_pin.setdefault(pincode, []).append(
                (row[2].strip(), row[3].strip(), row[5].strip(), lat, lon))
    if skipped:
        print("skipped %d unusable rows" % skipped, file=sys.stderr)
    return by_pin


def main() -> int:
    if not ensure_source():
        print("cannot proceed without %s" % SOURCE, file=sys.stderr)
        return 1

    by_pin = load(SOURCE)

    # Pass 1: collapse each pincode's places into one record.
    records: Dict[str, Dict[str, object]] = {}
    for pincode, entries in by_pin.items():
        places: List[str] = []
        seen = set()
        for place, _state, _district, _lat, _lon in entries:
            key = normalise(place)
            if key and key not in seen:
                seen.add(key)
                places.append(place)
        # State and district by majority vote: a handful of pincodes straddle a
        # boundary and GeoNames records both sides.
        state = statistics.mode([e[1] for e in entries])
        district = statistics.mode([e[2] for e in entries]) or state
        records[pincode] = {
            "state": state,
            "district": district,
            "places": places,
            "lat": round(statistics.fmean([e[3] for e in entries]), 5),
            "lon": round(statistics.fmean([e[4] for e in entries]), 5),
        }

    # Pass 2: measured density, then percentile thresholds for the tier cuts.
    centroids = {p: (float(r["lat"]), float(r["lon"]))
                 for p, r in records.items()}
    density = compute_density(centroids)
    ordered = sorted(density.values())

    def percentile(pct: int) -> int:
        idx = min(len(ordered) - 1, max(0, int(len(ordered) * pct / 100.0)))
        return ordered[idx]

    thresholds = {label: percentile(pct) for label, pct in TIER_CUTS}

    os.makedirs(OUT_DIR, exist_ok=True)
    tiers: Dict[str, int] = {}
    with gzip.open(OUT, "wt", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["pincode", "state", "district", "primary_place",
                         "lat", "lon", "tier", "density_10km", "n_places",
                         "places"])
        for pincode in sorted(records):
            rec = records[pincode]
            places = rec["places"]              # type: ignore[assignment]
            dens = density[pincode]
            tier = tier_from_density(dens, thresholds,
                                    str(rec["district"]))
            tiers[tier] = tiers.get(tier, 0) + 1
            writer.writerow([
                pincode, rec["state"], rec["district"],
                places[0] if places else "",
                "%.5f" % rec["lat"], "%.5f" % rec["lon"],
                tier, dens, len(places), "|".join(places),
            ])

    size = os.path.getsize(OUT)
    print("wrote %s" % os.path.relpath(OUT, REPO))
    print("  pincodes  : %d" % len(records))
    print("  size      : %.1f KiB gzipped" % (size / 1024.0))
    print("  density   : min %d  median %d  max %d"
          % (ordered[0], ordered[len(ordered) // 2], ordered[-1]))
    print("  cuts      : %s" % ", ".join(
        "%s>=%d" % (label, thresholds[label]) for label, _ in TIER_CUTS))
    print("  tiers     : %s" % ", ".join(
        "%s=%d" % kv for kv in sorted(tiers.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
