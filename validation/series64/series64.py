"""
Series 64 model 5s (Wellicome, Molland, Cic & Taunton 1999, Ship Science Report
106) as a `Hull`, from the body plan traced by `trace_bodyplan.py`.

Report 106, Table 1: L 1.6 m, L/B 12.8, B/T 2.0, L/vol^(1/3) 8.5, C_B 0.537,
C_P 0.633, C_M 0.848, static wetted area 0.261 m^2, LCB 6.4% aft of midships.
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from hydrosandbox.hull import Hull

_DATA = json.loads((Path(__file__).parent / "sections_5s.json").read_text())
Z_OVER_T = np.array(_DATA["z_over_T"])
SECTIONS = np.array(_DATA["half_breadth_over_half_beam"])  # the stem, then the 20 traced curves, bow to transom
# Station (of 20) of each curve. The report does not label them. The afterbody's ten are stations 11-20
# and the forebody's are a curve close to the stem, then 1-7, 9 and 10 (no station 8): with that
# reading the traced section areas and waterline beams follow the Series 64 parent's curves
# (Yeh 1965, Fig. 1) to within about 0.04, and with any other they do not.
STATIONS = np.array([0.0, 0.3, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) / 20.0
L, LB, BT, AREA = 1.6, 12.8, 2.0, 0.261
REPORT = dict(CB=0.537, CP=0.633, CM=0.848, slenderness=8.5, LCB_pct_aft=6.4)


def series64_hull(n_stations: int = 61, n_levels: int = 41, **hull_options):
    B = L / LB
    T = B / BT
    z = np.linspace(0.0, Z_OVER_T[-1], n_levels)
    table = np.array([np.interp(z, Z_OVER_T, s) for s in SECTIONS])
    x = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_stations)))
    half_breadths = np.array([PchipInterpolator(STATIONS, table[:, j])(x) for j in range(n_levels)]).T
    hull = Hull(x * L, (z - 1.0) * T, np.clip(half_breadths, 0.0, None) * B / 2, name="Series 64 5s", **hull_options)
    return hull, dict(L=L, B=B, T=T)


if __name__ == "__main__":
    hull, d = series64_hull()
    hs = hull.hydrostatics()
    vol, S, x_cb = (float(hs[k]) for k in ("volume", "wetted_surface", "x_cb"))
    zz = np.linspace(0, 1, 201)
    areas = np.array([2 * np.trapezoid(np.interp(zz, Z_OVER_T, s), zz) for s in SECTIONS]) * d["B"] / 2 * d["T"]
    print(f"C_B {vol / (d['L'] * d['B'] * d['T']):.3f} ({REPORT['CB']})   C_M {areas.max() / (d['B'] * d['T']):.3f} ({REPORT['CM']})   "
          f"C_P {vol / (areas.max() * d['L']):.3f} ({REPORT['CP']})")
    print(f"L/vol^(1/3) {d['L'] / vol ** (1 / 3):.2f} ({REPORT['slenderness']})   LCB {(x_cb / d['L'] - 0.5) * 100:+.1f}% aft ({REPORT['LCB_pct_aft']})   "
          f"wetted {S:.4f} m^2 ({AREA})")
    yeh = {5: (0.477, 0.568), 6: (0.579, 0.65), 7: (0.683, 0.73), 9: (0.85, 0.862), 10: (0.90, 0.905), 15: (0.785, 0.994),
           16: (0.672, 0.972), 17: (0.575, 0.945), 18: (0.499, 0.907), 19: (0.429, 0.866), 20: (0.381, 0.823)}
    print("station: traced A/Amax, B/Bmax | Series 64 parent A/Ax, B/Bx (read off Yeh 1965 Fig. 1, +/- 0.02)")
    for st, (a, b) in yeh.items():
        k = int(np.argmin(np.abs(STATIONS * 20 - st)))
        print(f"  {st:2d}: {areas[k] / areas.max():.2f}, {np.interp(1.0, Z_OVER_T, SECTIONS[k]):.2f} | {a:.2f}, {b:.2f}")
