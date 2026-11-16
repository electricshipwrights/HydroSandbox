"""
Is the bow wave that thin ship lacks the source of its trim-moment shortfall?

    uv run python validation/bow_wave/test.py

Bow-wave height and crest position from the analytical relations of Noblesse,
Delhommeau, Liu, Wan & Yang (2013), "Ship bow waves", J. Hydrodynamics 25(4),
eq. (6a, b), for a bow without rake or flare:

    Z_b g / U^2 = 2.2 tan(a) / (cos(a) (1 + Fr)),   X_b g / U^2 = 1.1 cos^8(a) / (1 + Fr),   Fr = U / sqrt(g D)

with `a` the waterline half entrance angle and `D` the draught at the bow (here
at the running attitude). A one-number load scale is built from them: the
weight of water in a wedge of height Z_b, length X_b and the hull's beam at the
crest, `rho g Z_b X_b (2 X_b tan a)`, acting at the crest. The shortfall of thin
ship's dynamic force and moment (measured - thin ship, as in `lift_2dt/test.py`)
is regressed on it. The slope is a shape-and-pressure factor, not expected to
be one; what matters is whether it is the SAME in every hull family, and how
well the scale correlates with the shortfall. Writes `summary.md`.
"""

import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lift_2dt"))
warnings.filterwarnings("ignore")
import test as lift_test  # builds `points`: the three families' shortfalls at the measured attitudes

G, RHO = 9.80665, lift_test.RHO


def bow_wave(hull, velocity, sinkage, trim, x_cg):
    x, z = hull.stations, hull.levels
    wl = np.array([np.interp(0.0, z, hb) for hb in hull.half_breadths])
    length = x[-1] - x[0]
    tan_a = np.interp(x[0] + 0.05 * length, x, wl) / (0.05 * length)
    cos_a = 1.0 / math.sqrt(1.0 + tan_a**2)
    keel_bow = z[np.argmax(hull.half_breadths[np.argmin(np.abs(x - (x[0] + 0.05 * length)))] > 1e-9)]
    draught = max(-keel_bow + sinkage - (x_cg - x[0]) * math.tan(math.radians(trim)), 0.2 * -keel_bow)
    fr = velocity / math.sqrt(G * draught)
    z_b = velocity**2 / G * 2.2 * tan_a / (cos_a * (1.0 + fr))
    x_b = velocity**2 / G * 1.1 * cos_a**8 / (1.0 + fr)
    return z_b, x_b, tan_a


for p in lift_test.points:
    v = p["Fn"] * math.sqrt(G * p["L"])
    z_b, x_b, tan_a = bow_wave(p["hull"], v, p["sinkage"], p["trim"], p["x_cg"])
    load = RHO * G * z_b * x_b * 2.0 * x_b * tan_a * p["n"]
    p["bw_force"] = load / p["weight"]
    p["bw_moment"] = load * (p["x_cg"] - p["hull"].stations[0] - x_b) / (p["weight"] * p["L"])
    p["zb_over_T"] = z_b / -p["hull"].levels[0]

rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
out = ["# Bow-wave load scale against thin ship's shortfall", "",
       "Fn >= 0.4. `slope`: least-squares slope of the shortfall on the load scale (through the origin); `r`: correlation; "
       "then the rms shortfall before and after subtracting slope x scale, with the slope fitted on that family and with "
       "the NPL family's slope. Force in % of weight, moment in 1e-3 weight x length.", "",
       "| family | n | moment: slope, r | moment rms: thin -> own slope -> NPL slope | force: slope, r | force rms: thin -> own slope -> NPL slope |",
       "|---|---|---|---|---|---|"]
npl = {}
for family in ("Southampton NPL", "Series 64 5s", "Athena"):
    sel = [p for p in lift_test.points if p["family"] == family and p["Fn"] >= 0.4 - 1e-9]
    cells = []
    for q, s in (("moment", 1000.0), ("force", 100.0)):
        y = s * np.array([p[f"d_{q}"] for p in sel]); x = s * np.array([p[f"bw_{q}"] for p in sel])
        slope = (x @ y) / (x @ x)
        npl.setdefault(q, slope)
        cells += [f"{slope:.2f}, {np.corrcoef(x, y)[0, 1]:+.2f}", f"{rms(y):.1f} -> {rms(y - slope * x):.1f} -> {rms(y - npl[q] * x):.1f}"]
    out.append(f"| {family} | {len(sel)} | " + " | ".join(cells) + " |")
out += ["", "## Southampton monohulls by model, Fn 0.65-1.00", "",
        "| model | moment shortfall (median) | load-scale moment x NPL slope | bow-wave height / draught (formula) |", "|---|---|---|---|"]
from npl import MODELS
for model in MODELS:
    sel = [p for p in lift_test.points if p["family"] == "Southampton NPL" and p["group"] == model and p["n"] == 1 and p["Fn"] >= 0.65 - 1e-9]
    med = lambda k: float(np.median([p[k] for p in sel]))
    out.append(f"| {model} | {1e3 * med('d_moment'):+.1f} | {1e3 * npl['moment'] * med('bw_moment'):+.1f} | {med('zb_over_T'):.2f} |")
(HERE / "summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
