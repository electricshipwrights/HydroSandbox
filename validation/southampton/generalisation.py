"""
How far does the Phase A correction reach beyond what it was fitted on?

    uv run python validation/southampton/generalisation.py

Open loop (no equilibrium solves): the correction is refitted with a whole
group of Report 71 points held out, and scored on that group, for groups chosen
to force extrapolation in one direction at a time. Then the correction fitted on
all of Report 71 is applied to R/V Athena, a different hull family. Errors are
in the dynamic force / weight x 100 and moment / (weight x length) x 1000.
Writes `report/generalisation.md`.
"""

import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "athena"))
warnings.filterwarnings("ignore")
from npl import MODELS
from phase_a import CLOSURE, RIDGE, SPACING, features

G = 9.80665
rows = [r for r in json.loads((HERE / "attitude.json").read_text()) if r["closure"] == CLOSURE]
feats = [features(r["Fn"], MODELS[r["model"]]["slenderness"], MODELS[r["model"]]["BT"], SPACING[r["config"]],
                  r["force_model"], r["moment_model"]) for r in rows]
errs = {q: np.array([r[f"{q}_measured"] - r[f"{q}_model"] for r in rows]) for q in ("force", "moment")}
scale = {"force": 100.0, "moment": 1000.0}


def fit(keep):
    out = []
    for k, q in enumerate(("force", "moment")):
        a = np.array([f[k] for f in feats])[keep]
        out.append(np.linalg.solve(a.T @ a + RIDGE * 1e-4 * np.eye(a.shape[1]), a.T @ errs[q][keep]))
    return out


def held_out(test):
    w = fit(~test)
    cells = []
    for k, q in enumerate(("force", "moment")):
        a = np.array([f[k] for f in feats])[test]
        before, after = scale[q] * errs[q][test], scale[q] * (errs[q][test] - a @ w[k])
        cells += [f"{np.sqrt(np.mean(before**2)):.1f}", f"{np.sqrt(np.mean(after**2)):.1f}"]
    return cells


name = np.array([r["model"] for r in rows])
config = np.array([r["config"] for r in rows])
fn = np.array([r["Fn"] for r in rows])
tests = [
    ("Each model in turn (the Phase A test), pooled", None),
    ("Fullest group: 3b (L/vol^(1/3) 6.3)", name == "3b"),
    ("Models 4 (7.4)", np.char.startswith(name, "4")),
    ("Models 5 (8.5)", np.char.startswith(name, "5")),
    ("Most slender group: models 6 (9.5)", np.char.startswith(name, "6")),
    ("Narrowest: all B/T 1.5 (a)", np.char.endswith(name, "a")),
    ("All B/T 2.0 (b)", np.char.endswith(name, "b")),
    ("Widest: all B/T 2.5 (c)", np.char.endswith(name, "c")),
    ("All catamarans (fitted on monohulls only)", config != "mono"),
    ("Closest spacing S/L 0.2", config == "s0.2"),
    ("High speed Fn >= 0.8 (fitted below)", fn >= 0.8 - 1e-9),
    ("Low speed Fn <= 0.35 (fitted above)", fn <= 0.35 + 1e-9),
]
out = [f"# How far the Phase A correction generalises (closure `{CLOSURE}`, open loop)", "",
       "rms error of the dynamic force [% of weight] and trim moment [1e-3 weight x length] on the held-out "
       "group: thin ship, then thin ship + the correction fitted without that group.", "",
       "| held out | n | force: thin ship | force: corrected | moment: thin ship | moment: corrected |", "|---|---|---|---|---|---|"]
for label, test in tests:
    if test is None:
        after = {q: np.zeros(len(rows)) for q in errs}
        for model in MODELS:
            t = name == model
            w = fit(~t)
            for k, q in enumerate(("force", "moment")):
                after[q][t] = errs[q][t] - np.array([f[k] for f in feats])[t] @ w[k]
        cells = [f"{scale[q] * np.sqrt(np.mean(v**2)):.1f}" for q in ("force", "moment") for v in (errs[q], after[q])]
        out.append(f"| {label} | {len(rows)} | " + " | ".join(cells) + " |")
    else:
        out.append(f"| {label} | {int(test.sum())} | " + " | ".join(held_out(test)) + " |")

### Athena: a different family, with the correction fitted on all of Report 71.
import hydrosandbox as hsb
from hydrosandbox.hull import CraftBuildup
from athena import LPP, athena_hull

w_all = fit(np.ones(len(rows), bool))
hull = athena_hull()
statics = hull.hydrostatics()
volume, x_cb = float(statics["volume"]), float(statics["x_cb"])
half_beam = float(np.interp(0.0, hull.levels, hull.half_breadths[-1]))
water = hsb.Water.freshwater(temperature_C=15.0)
weight = volume * float(water.density()) * G
slenderness, bt = LPP / volume ** (1 / 3), 0.1332 * LPP / 0.183
hull.transom_closure = 6.0 * half_beam
out += ["", f"## R/V Athena (L/vol^(1/3) {slenderness:.2f}, B/T {bt:.1f}; Report 71 spans 6.3-9.5 and 1.5-2.5)", "",
        "The correction fitted on all of Report 71, applied unchanged. Measured values are inferred from the measured "
        "attitude as for Report 71; Athena's tow point is not given and is taken at the centre of gravity.", "",
        "| Fn | force: measured | thin ship | corrected | moment: measured | thin ship | corrected |", "|---|---|---|---|---|---|---|"]
acc = {k: [] for k in ("ft", "fc", "mt", "mc")}
for r in csv.DictReader(l for l in open(HERE.parent / "athena" / "jenkins1984.csv") if not l.startswith("#")):
    f = float(r["Fn"])
    trim = math.degrees(math.atan(float(r["Ctau_e2"]) / 100))
    sinkage = -float(r["Cs_e2"]) / 100 * LPP + (x_cb - 0.5 * LPP) * math.tan(math.radians(trim))
    res = CraftBuildup(hull=hull, velocity=f * math.sqrt(G * LPP), mass=weight / G, xyz_cg=[x_cb, 0.0, 0.0], water=water,
                       sinkage=sinkage, trim=trim, wave_length_scales=hull.default_wave_length_scales(None),
                       include_interference_breakdown=False, include_transom_drag=True).run()
    v = lambda key: float(np.ravel(res[key])[0])
    f_thin, m_thin = v("F_z_hull_dynamic") / weight, v("M_y_hull_dynamic") / (weight * LPP)
    f_meas, m_meas = f_thin - v("F_z_net") / weight, m_thin - v("M_y_net") / (weight * LPP)
    ff, mf = features(f, slenderness, bt, None, f_thin, m_thin)
    f_corr, m_corr = f_thin + ff @ w_all[0], m_thin + mf @ w_all[1]
    out.append(f"| {f:.2f} | {100 * f_meas:+.1f} | {100 * f_thin:+.1f} | {100 * f_corr:+.1f} | {1e3 * m_meas:+.1f} | {1e3 * m_thin:+.1f} | {1e3 * m_corr:+.1f} |")
    for k, val in (("ft", 100 * (f_thin - f_meas)), ("fc", 100 * (f_corr - f_meas)), ("mt", 1e3 * (m_thin - m_meas)), ("mc", 1e3 * (m_corr - m_meas))):
        acc[k].append(val)
rms = lambda k: np.sqrt(np.mean(np.square(acc[k])))
out += ["", f"rms error: force {rms('ft'):.1f} thin ship, {rms('fc'):.1f} corrected; moment {rms('mt'):.1f} thin ship, {rms('mc'):.1f} corrected."]
(HERE / "report" / "generalisation.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
