"""
The hull's dynamic sinkage force and trim moment, model against measurement.

    uv run python validation/southampton/attitude.py [--closures auto,none,c6] [--jobs 6]

A model towed free to sink and trim settles where buoyancy, weight and the
hull's dynamic (speed-dependent) force and moment balance. So the measured
attitude gives the *measured* dynamic force and moment directly: they are
minus the hydrostatic imbalance at that attitude (the tow force acts through
the centre of gravity). This evaluates the model's thin-ship dynamic force and
moment at each measured attitude and compares, with no equilibrium solve.

Force is given as a fraction of the weight (positive up), moment as a fraction
of weight x length (positive bow up). Writes `attitude.json` and
`report/attitude.md`.
"""

import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
G = 9.80665
SPACING = {"mono": None, "s0.2": 0.2, "s0.3": 0.3, "s0.4": 0.4, "s0.5": 0.5}


def model_cases(model: str, closure: str) -> list:
    warnings.filterwarnings("ignore")
    import hydrosandbox as hsb
    from hydrosandbox.hull import CraftBuildup
    from npl import MODELS, npl_hull
    from spectra import closure_value

    water = hsb.Water.freshwater(temperature_C=15.0)
    rho = float(water.density())
    hull, d = npl_hull(model)
    statics = hull.hydrostatics()
    volume, x_cb = float(statics["volume"]), float(statics["x_cb"])
    transom_half_beam = float(np.interp(0.0, hull.levels, hull.half_breadths[-1]))
    rows = []
    for r in csv.DictReader(open(HERE / "r71_attitude.csv")):
        if r["model"] != model or r["flag"]:
            continue
        fn, config = float(r["Fn"]), r["config"]
        n_hulls = 1 if SPACING[config] is None else 2
        spacing = None if SPACING[config] is None else SPACING[config] * d["L"]
        hull.transom_closure = closure_value(closure, fn, MODELS[model]["BT"], transom_half_beam)
        weight = n_hulls * volume * rho * G
        res = CraftBuildup(
            hull=hull, velocity=fn * math.sqrt(G * d["L"]), mass=weight / G,
            xyz_cg=[x_cb, 0.0, -2.0 * d["T"] / 3.0], water=water, demihull_spacing=spacing,
            sinkage=float(r["sinkage_pct_draught"]) / 100 * d["T"], trim=float(r["trim_deg"]),
            wave_length_scales=hull.default_wave_length_scales(spacing),
            include_interference_breakdown=False, include_transom_drag=True,
        ).run()
        value = lambda key: float(np.ravel(res[key])[0])
        f_model, m_model = value("F_z_hull_dynamic"), value("M_y_hull_dynamic")
        rows.append(dict(
            model=model, config=config, closure=closure, Fn=fn,
            force_model=f_model / weight, force_measured=(f_model - value("F_z_net")) / weight,
            moment_model=m_model / (weight * d["L"]),
            moment_measured=(m_model - value("M_y_net")) / (weight * d["L"]),
        ))
    return rows


def report(rows):
    from npl import MODELS

    closures = sorted({r["closure"] for r in rows})
    bands = [("0.20-0.35", 0.2, 0.35), ("0.40-0.60", 0.4, 0.6), ("0.65-1.00", 0.65, 1.0)]
    out = ["# Dynamic sinkage force and trim moment: thin ship against Report 71", "",
           "Measured values are minus the hydrostatic imbalance at the measured attitude. Force / weight x 100 "
           "(positive up); moment / (weight x length) x 1000 (positive bow up). Each cell: median measured, "
           "median model, rms difference.", ""]
    for quantity, scale in (("force", 100.0), ("moment", 1000.0)):
        for group, pick in (("monohull", lambda r: r["config"] == "mono"), ("catamaran", lambda r: r["config"] != "mono")):
            out += [f"## {quantity.capitalize()}, {group}", "",
                    "| closure | " + " | ".join(f"Fn {b[0]}" for b in bands) + " |", "|---|---|---|---|"]
            for closure in closures:
                cells = []
                for _, lo, hi in bands:
                    sel = [r for r in rows if r["closure"] == closure and pick(r) and lo - 1e-9 <= r["Fn"] <= hi + 1e-9]
                    a = scale * np.array([r[f"{quantity}_measured"] for r in sel])
                    b = scale * np.array([r[f"{quantity}_model"] for r in sel])
                    cells.append(f"{np.median(a):+.2f} / {np.median(b):+.2f} / {np.sqrt(np.mean((b - a) ** 2)):.2f}")
                out.append(f"| {closure} | " + " | ".join(cells) + " |")
            out.append("")
    out += ["## Monohull by model, Fn 0.65-1.00, first closure listed", "",
            "| model | L/vol^(1/3) | B/T | force: measured / model | moment: measured / model |", "|---|---|---|---|---|"]
    for model, m in MODELS.items():
        sel = [r for r in rows if r["closure"] == closures[0] and r["model"] == model and r["config"] == "mono" and r["Fn"] >= 0.65 - 1e-9]
        med = lambda key, s: s * np.median([r[key] for r in sel])
        out.append(f"| {model} | {m['slenderness']} | {m['BT']} | {med('force_measured', 100):+.2f} / {med('force_model', 100):+.2f} "
                   f"| {med('moment_measured', 1000):+.2f} / {med('moment_model', 1000):+.2f} |")
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "attitude.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    import hydrosandbox as hsb
    from npl import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--closures", default="auto,none,c6")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    path = HERE / "attitude.json"
    if not args.report_only:
        tasks = [(m, c) for c in args.closures.split(",") for m in MODELS]
        rows = [r for part in hsb.parallel_map(model_cases, tasks, jobs=args.jobs, progress=True) for r in part]
        path.write_text(json.dumps(rows))
    report(json.loads(path.read_text()))


def learnability(rows):
    """Leave-one-model-out ridge fit of (measured - model) on smooth features of Fn, form and spacing."""
    from npl import MODELS

    lines = ["", "## Is the error learnable? Leave-one-model-out correction", "",
             "A linear correction in smooth features of Fn, L/vol^(1/3), B/T and 1/spacing, fitted without the "
             "held-out model. rms error of force / weight x 100 and moment / (weight x length) x 1000, Fn >= 0.3, "
             "monohull and catamaran together.", "",
             "| closure | force: thin ship | force: corrected | moment: thin ship | moment: corrected |", "|---|---|---|---|---|"]
    for closure in sorted({r["closure"] for r in rows}):
        sel = [r for r in rows if r["closure"] == closure and r["Fn"] >= 0.3 - 1e-9]
        x = np.array([(r["Fn"] - 0.65) / 0.35 for r in sel])
        s = np.array([(MODELS[r["model"]]["slenderness"] - 8.0) / 1.6 for r in sel])
        b = np.array([(MODELS[r["model"]]["BT"] - 2.0) / 0.5 for r in sel])
        c = np.array([0.0 if r["config"] == "mono" else 0.2 / float(r["config"][1:]) for r in sel])
        feats = np.stack([np.ones_like(x), x, x**2, x**3, s, b, c, x * s, x * b, x * c, x**2 * s, x**2 * c, s * c], axis=1)
        names = np.array([r["model"] for r in sel])
        cells = []
        for quantity, scale in (("force", 100.0), ("moment", 1000.0)):
            err = scale * np.array([r[f"{quantity}_measured"] - r[f"{quantity}_model"] for r in sel])
            held = np.zeros_like(err)
            for model in MODELS:
                test = names == model
                a = feats[~test]
                w = np.linalg.solve(a.T @ a + 0.1 * np.eye(a.shape[1]), a.T @ err[~test])
                held[test] = err[test] - feats[test] @ w
            cells += [f"{np.sqrt(np.mean(err**2)):.2f}", f"{np.sqrt(np.mean(held**2)):.2f}"]
        lines.append(f"| {closure} | " + " | ".join(cells) + " |")
    with open(HERE / "report" / "attitude.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    learnability(json.loads((HERE / "attitude.json").read_text()))
