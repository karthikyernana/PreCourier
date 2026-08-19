# Data sources

Every external dataset this system depends on, with its licence and the script
that derives our copy from it. PRD section 13: *no external statistic in any
deliverable without a citation and a date.*

---

## `pincodes.csv.gz` — India postal-code reference

| | |
|---|---|
| **Source** | GeoNames postal-code export for India, `https://download.geonames.org/export/zip/IN.zip` |
| **Licence** | Creative Commons Attribution 4.0 — <https://creativecommons.org/licenses/by/4.0/> |
| **Retrieved** | 2026-08-29 (upstream file dated 2026-08-29) |
| **Upstream rows** | 155,570 place records |
| **Our rows** | 19,238 pincodes, one per row |
| **Derived by** | [`tools/build_pincode_reference.py`](../../tools/build_pincode_reference.py) |

> This product includes data created by GeoNames (<https://www.geonames.org/>),
> licensed under CC BY 4.0. GeoNames provides the data "as is" without warranty
> or any representation of accuracy, timeliness or completeness.

### Rebuild

```bash
python3 tools/build_pincode_reference.py
```

Downloads the upstream zip into `var/raw/` if absent, then rewrites the table.
Requires no third-party packages.

### What each column is for

| Column | Used by |
|---|---|
| `pincode` | `pincode_valid` feature; `pincode_unknown` verification check |
| `state`, `district` | `pincode_city_consistent` feature; `pincode_city_mismatch` verification check |
| `primary_place` | Human-readable label in the console |
| `lat`, `lon` | Distance-band feature, measured from the merchant's origin pincode |
| `tier` | Geography feature; simulated-carrier serviceability |
| `density_10km` | Geography feature — the numeric form of `tier`, preferred by the model |
| `n_places`, `places` | `locality_in_reference` feature; `locality_dropped` verification check |

### Two honest caveats

**`tier` and `density_10km` are derived, not authoritative.** India has no
machine-readable official city-tier list; the RBI and HRA classifications are
population-based and revised on a census cycle. We measure how many pincode
centroids fall within 10 km and cut at declared percentiles (see the module
docstring). This replaced an earlier district-name-matching approach that
mislabelled Connaught Place as rural and missed Gurgaon on a spelling variant —
name matching cannot be made complete and its failures are silent.

**COD serviceability is not in this dataset.** Real serviceability is
carrier-specific and comes from a carrier contract we do not have. The simulated
carrier derives it deterministically from `tier` and `pincode`, and every surface
that displays a serviceability verdict labels it as simulated.
