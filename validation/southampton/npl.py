"""
The Southampton catamaran-series demihulls (Molland, Wellicome & Couser 1994,
Ship Science Report 71) as `Hull` offset tables.

All ten models are the NPL round-bilge parent (Bailey 1976, model 100A) scaled
in beam and draught. The parent's body plan is in `npl_sections.json`: eleven
equally spaced sections, 0 (transom) to 10 (stem at the design waterline), each
as half-breadth / (B/2) against height above base / T. It was digitised from the
vector body plan in Robards (2008), "The Hydrodynamics of High-Speed
Transom-Stern Vessels", UNSW, Fig. A.13.
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from hydrosandbox.hull import Hull

_DATA = json.loads((Path(__file__).parent / "npl_sections.json").read_text())
Z_OVER_T = np.array(_DATA["z_over_T"])
SECTIONS = np.array(_DATA["half_breadth_over_half_beam"])  # (11 sections, n_z), transom first

LENGTH = 1.6  # [m], all models
# Report 71, Table II: L/B, B/T, L/vol^(1/3), static wetted area A [m^2] of one demihull.
MODELS = {
    "3b": dict(LB=7.0, BT=2.0, slenderness=6.27, area=0.434),
    "4a": dict(LB=10.4, BT=1.5, slenderness=7.40, area=0.348),
    "4b": dict(LB=9.0, BT=2.0, slenderness=7.41, area=0.338),
    "4c": dict(LB=8.0, BT=2.5, slenderness=7.39, area=0.340),
    "5a": dict(LB=12.8, BT=1.5, slenderness=8.51, area=0.282),
    "5b": dict(LB=11.0, BT=2.0, slenderness=8.50, area=0.276),
    "5c": dict(LB=9.9, BT=2.5, slenderness=8.49, area=0.277),
    "6a": dict(LB=15.1, BT=1.5, slenderness=9.50, area=0.240),
    "6b": dict(LB=13.1, BT=2.0, slenderness=9.50, area=0.233),
    "6c": dict(LB=11.7, BT=2.5, slenderness=9.50, area=0.234),
}
# Report 71, Table II: the same for every model.
CB_REPORT, CP_REPORT, LCB_REPORT = 0.397, 0.693, 6.4  # LCB in % of L aft of midships


def npl_hull(model: str, n_stations: int = 61, n_levels: int = 41, **hull_options):
    """One demihull of `model`, and its `L`, `B`, `T` [m]. x runs from the stem (0) aft."""
    L = LENGTH
    B = L / MODELS[model]["LB"]
    T = B / MODELS[model]["BT"]
    z = np.linspace(0.0, Z_OVER_T[-1], n_levels)
    x_sections = np.linspace(1.0, 0.0, 11)  # section 0 is the transom, x/L = 1
    table = np.array([np.interp(z, Z_OVER_T, s) for s in SECTIONS])
    x = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_stations)))
    half_breadths = np.array(
        [PchipInterpolator(x_sections[::-1], table[::-1, j])(x) for j in range(n_levels)]
    ).T
    hull = Hull(
        x * L, (z - 1.0) * T, np.clip(half_breadths, 0.0, None) * B / 2,
        name=f"NPL {model}", **hull_options,
    )
    return hull, dict(L=L, B=B, T=T)


if __name__ == "__main__":
    print("model   CB (0.397)   LCB %L aft (6.4)   S [m^2]  report A   S/A")
    for model, m in MODELS.items():
        hull, d = npl_hull(model)
        hs = hull.hydrostatics()
        vol, S, x_cb = (float(hs[k]) for k in ("volume", "wetted_surface", "x_cb"))
        print(
            f"{model}     {vol / (d['L'] * d['B'] * d['T']):.3f}        "
            f"{(x_cb / d['L'] - 0.5) * 100:+.1f}              {S:.4f}   {m['area']:.3f}     {S / m['area']:.3f}"
            f"   L/vol^(1/3) {d['L'] / vol ** (1 / 3):.2f} ({m['slenderness']})"
        )
