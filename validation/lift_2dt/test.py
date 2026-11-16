"""
Does 2D+t dynamic lift account for what thin ship misses in the hull's dynamic
force and trim moment? Tested at the measured attitudes of three hull families.

    uv run python validation/lift_2dt/test.py

The "shortfall" is the measured dynamic force or moment (inferred from the
measured attitude, as in `southampton/attitude.py`) less thin ship's, with the
six-half-breadth transom closure. If the 2D+t lift is the missing physics, the
shortfall regressed on it has a slope near one in every family, and adding it
cuts the error. Force in % of weight, moment in 1e-3 weight x length. Writes
`summary.md`.
"""

import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
for sub in ("southampton", "series64", "athena", "lift_2dt"):
    sys.path.insert(0, str(ROOT / sub))
warnings.filterwarnings("ignore")
import hydrosandbox as hsb
from hydrosandbox.hull import CraftBuildup
from athena import LPP, athena_hull
from lift import lift_2dt
from npl import MODELS, npl_hull
from series64 import series64_hull

G = 9.80665
water = hsb.Water.freshwater(temperature_C=15.0)
RHO = float(water.density())
points = []  # dict(family, group, Fn, hull, n, L, sinkage, trim, pivot, x_cg, weight, d_force, d_moment)


def add(family, group, hull, n_hulls, L, fn, sinkage, trim, z_cg, d_force, d_moment):
    st = hull.hydrostatics()
    x_cb, weight = float(st["x_cb"]), n_hulls * float(st["volume"]) * RHO * G
    points.append(dict(family=family, group=group, Fn=fn, hull=hull, n=n_hulls, L=L, sinkage=sinkage, trim=trim,
                       pivot=(x_cb, z_cg), x_cg=x_cb, weight=weight, d_force=d_force, d_moment=d_moment))


### Southampton (Report 71)
hulls = {m: npl_hull(m) for m in MODELS}
attitude = {(r["model"], r["config"], float(r["Fn"])): r for r in csv.DictReader(open(ROOT / "southampton" / "r71_attitude.csv")) if not r["flag"]}
for r in json.loads((ROOT / "southampton" / "attitude.json").read_text()):
    if r["closure"] != "c6":
        continue
    hull, d = hulls[r["model"]]
    a = attitude[(r["model"], r["config"], r["Fn"])]
    add("Southampton NPL", r["model"], hull, 1 if r["config"] == "mono" else 2, d["L"], r["Fn"],
        float(a["sinkage_pct_draught"]) / 100 * d["T"], float(a["trim_deg"]), -2.0 * d["T"] / 3.0,
        r["force_measured"] - r["force_model"], r["moment_measured"] - r["moment_model"])

### Series 64 model 5s (Report 106)
hull, d = series64_hull()
for r in json.loads((ROOT / "series64" / "results.json").read_text()):
    add("Series 64 5s", r["config"], hull, 1 if r["config"] == "mono" else 2, d["L"], r["Fn"],
        r["sinkage_measured"] / 100 * d["T"], r["trim_measured"], -2.0 * d["T"] / 3.0,
        r["force_measured"] - r["force_thin"], r["moment_measured"] - r["moment_thin"])

### Athena (Jenkins 1984), free to sink and trim
hull = athena_hull()
st = hull.hydrostatics()
x_cb, weight = float(st["x_cb"]), float(st["volume"]) * RHO * G
hull.transom_closure = 6.0 * float(np.interp(0.0, hull.levels, hull.half_breadths[-1]))
for r in csv.DictReader(l for l in open(ROOT / "athena" / "jenkins1984.csv") if not l.startswith("#")):
    fn = float(r["Fn"])
    trim = math.degrees(math.atan(float(r["Ctau_e2"]) / 100))
    sinkage = -float(r["Cs_e2"]) / 100 * LPP + (x_cb - 0.5 * LPP) * math.tan(math.radians(trim))
    res = CraftBuildup(hull=hull, velocity=fn * math.sqrt(G * LPP), mass=weight / G, xyz_cg=[x_cb, 0.0, 0.0], water=water,
                       sinkage=sinkage, trim=trim, wave_length_scales=hull.default_wave_length_scales(None),
                       include_interference_breakdown=False, include_transom_drag=True).run()
    add("Athena", "mono", hull, 1, LPP, fn, sinkage, trim, 0.0,
        -float(np.ravel(res["F_z_net"])[0]) / weight, -float(np.ravel(res["M_y_net"])[0]) / (weight * LPP))

VARIANTS = {"trim": dict(alpha_model="trim"), "run": dict(alpha_model="run"),
            "trim, no transom fade": dict(alpha_model="trim", transom_reduction=False),
            "trim, no fade, no cross-flow drag": dict(alpha_model="trim", transom_reduction=False, crossflow_cd=0.0)}
for p in points:
    for model, kw in VARIANTS.items():
        f, m = lift_2dt(p["hull"], p["Fn"] * math.sqrt(G * p["L"]), RHO, p["sinkage"], p["trim"], p["pivot"], p["x_cg"], **kw)
        p[f"f_{model}"], p[f"m_{model}"] = p["n"] * f / p["weight"], p["n"] * m / (p["weight"] * p["L"])

rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
out = ["# 2D+t dynamic lift against thin ship's shortfall", "",
       "Shortfall = measured - thin ship, at the measured attitude. `slope` is the least-squares slope of the shortfall "
       "on the 2D+t prediction (through the origin; 1.0 means the term is the missing physics); `r` their correlation; "
       "then the rms error of thin ship alone and of thin ship + 2D+t. Force in % of weight, moment in 1e-3 weight x length.", ""]
for lo, hi, label in ((0.4, 1.01, "Fn 0.40-1.00"), (0.2, 0.36, "Fn 0.20-0.35")):
    out += [f"## {label}", "", "| family | n | variant | force: slope, r | force rms: thin -> +2D+t | moment: slope, r | moment rms: thin -> +2D+t |",
            "|---|---|---|---|---|---|---|"]
    for family in ("Southampton NPL", "Series 64 5s", "Athena"):
        sel = [p for p in points if p["family"] == family and lo - 1e-9 <= p["Fn"] <= hi]
        for model in VARIANTS:
            cells = []
            for q, s, key in (("force", 100.0, "f"), ("moment", 1000.0, "m")):
                y = s * np.array([p[f"d_{q}"] for p in sel]); x = s * np.array([p[f"{key}_{model}"] for p in sel])
                cells += [f"{(x @ y) / (x @ x):.2f}, {np.corrcoef(x, y)[0, 1]:+.2f}", f"{rms(y):.1f} -> {rms(y - x):.1f}"]
            out.append(f"| {family} | {len(sel)} | {model} | " + " | ".join(cells) + " |")
    out.append("")
out += ["## Southampton monohulls by model, Fn 0.65-1.00, angle of attack = trim", "",
        "| model | moment shortfall (median) | 2D+t moment (median) | force shortfall | 2D+t force |", "|---|---|---|---|---|"]
for model in MODELS:
    sel = [p for p in points if p["family"] == "Southampton NPL" and p["group"] == model and p["n"] == 1 and p["Fn"] >= 0.65 - 1e-9]
    med = lambda k, s: s * float(np.median([p[k] for p in sel]))
    out.append(f"| {model} | {med('d_moment', 1e3):+.1f} | {med('m_trim', 1e3):+.1f} | {med('d_force', 100):+.1f} | {med('f_trim', 100):+.1f} |")
(HERE / "summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
