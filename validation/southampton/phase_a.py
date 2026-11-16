"""
Phase A of the surrogate plan: a learned correction to the hull's thin-ship
dynamic sinkage force and trim moment, fitted to Report 71 and tested closed
loop with each hull held out.

    uv run python validation/southampton/phase_a.py fit        (needs attitude.json; seconds)
    uv run python validation/southampton/phase_a.py solve [--jobs 18]
    uv run python validation/southampton/phase_a.py report

`fit` learns, by ridge regression, additive corrections to force / weight and
moment / (weight x length) from the labels `attitude.py` infers at the measured
attitudes, once on all hulls and once per held-out hull. `solve` then finds the
free running attitude of every model, configuration and speed twice: with plain
thin ship, and with the correction fitted *without that model*. `report`
compares both with the measured trim and sinkage.

The correction's inputs are Froude number, volume / L^3, B/T, demihull spacing
and (for the moment) the thin-ship force and moment themselves, so it follows
the attitude through them. It is specific to the transom closure it was fitted
with (`CLOSURE`).
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
CLOSURE = "c6"
RIDGE = 0.3
SPACING = {"mono": None, "s0.2": 0.2, "s0.3": 0.3, "s0.4": 0.4, "s0.5": 0.5}
FROUDE = [round(0.2 + 0.05 * i, 2) for i in range(17)]


def features(fn, slenderness, bt, spacing_ratio, force_thin, moment_thin):
    """Feature vectors for the force and the moment correction (all inputs scalars)."""
    x = (fn - 0.6) / 0.4
    s = (slenderness - 8.0) / 1.6
    v = (1.0 / slenderness**3 - 0.0022) / 0.0015  # volume / L^3
    b = (bt - 2.0) / 0.5
    c = 0.0 if spacing_ratio is None else 0.2 / spacing_ratio
    powers = [1.0, x, x**2, x**3, x**4]

    def base(f):
        return powers + [p * f for p in powers] + [p * b for p in powers[:3]] + [p * c for p in powers] \
            + [f * c, f * c * x, b * c]

    fm, mm = 10.0 * force_thin, 20.0 * moment_thin
    return np.array(base(s)), np.array(base(v) + [fm, mm, fm * v, mm * v, mm * x, fm * x])


def fit():
    from npl import MODELS

    rows = [r for r in json.loads((HERE / "attitude.json").read_text()) if r["closure"] == CLOSURE]
    feats = [features(r["Fn"], MODELS[r["model"]]["slenderness"], MODELS[r["model"]]["BT"],
                      SPACING[r["config"]], r["force_model"], r["moment_model"]) for r in rows]
    names = np.array([r["model"] for r in rows])
    weights = {}
    for held in [None] + list(MODELS):
        keep = names != held if held else np.ones(len(rows), bool)
        w = []
        for k, quantity in enumerate(("force", "moment")):
            a = np.array([f[k] for f in feats])[keep]
            err = np.array([r[f"{quantity}_measured"] - r[f"{quantity}_model"] for r in rows])[keep]
            w.append(np.linalg.solve(a.T @ a + RIDGE * 1e-4 * np.eye(a.shape[1]), a.T @ err).tolist())
        weights[held or "all"] = w
    (HERE / "phase_a_weights.json").write_text(json.dumps(dict(closure=CLOSURE, weights=weights)))
    print("fitted on", len(rows), "points; closure", CLOSURE)


class CorrectedDynamics:
    """Stands in for `CraftBuildup(hull_dynamics=...)`: thin ship plus the learned correction."""

    def __init__(self, hull, velocity, density, weight, length, slenderness, bt, spacing_ratio, weights):
        self.hull, self.velocity, self.density = hull, velocity, density
        self.weight, self.length = weight, length
        self.slenderness, self.bt, self.spacing_ratio = slenderness, bt, spacing_ratio
        self.w_force, self.w_moment = (np.array(w) for w in weights) if weights else (None, None)
        self.fn = velocity / math.sqrt(G * length)

    def __call__(self, sinkage, trim, pivot_xz=(0.0, 0.0), x_ref=None):
        spacing = None if self.spacing_ratio is None else self.spacing_ratio * self.length
        thin = self.hull.dynamic_force(self.velocity, self.density, G, sinkage, trim, pivot_xz, x_ref=x_ref,
                                       demihull_spacing=spacing)
        force, moment = float(np.ravel(thin["force_up"])[0]), float(np.ravel(thin["moment_bow_up"])[0])
        if self.w_force is not None:
            f_feat, m_feat = features(self.fn, self.slenderness, self.bt, self.spacing_ratio,
                                      force / self.weight, moment / (self.weight * self.length))
            force += self.weight * float(f_feat @ self.w_force)
            moment += self.weight * self.length * float(m_feat @ self.w_moment)
        return dict(thin, force_up=force, moment_bow_up=moment)


def solve_curve(model: str, config: str, corrected: bool) -> list:
    warnings.filterwarnings("ignore")
    import hydrosandbox as hsb
    from hydrosandbox.hull import CraftBuildup
    from npl import MODELS, npl_hull
    from spectra import closure_value

    weights = json.loads((HERE / "phase_a_weights.json").read_text())["weights"][model] if corrected else None
    water = hsb.Water.freshwater(temperature_C=15.0)
    rho = float(water.density())
    hull, d = npl_hull(model)
    statics = hull.hydrostatics()
    volume, x_cb = float(statics["volume"]), float(statics["x_cb"])
    transom_half_beam = float(np.interp(0.0, hull.levels, hull.half_breadths[-1]))
    n_hulls = 1 if SPACING[config] is None else 2
    spacing = None if SPACING[config] is None else SPACING[config] * d["L"]
    weight = n_hulls * volume * rho * G
    rows, guess = [], (0.0, 0.0)
    for fn in FROUDE:
        v = fn * math.sqrt(G * d["L"])
        hull.transom_closure = closure_value(CLOSURE, fn, MODELS[model]["BT"], transom_half_beam)
        dynamics = CorrectedDynamics(hull, v, rho, weight, d["L"], MODELS[model]["slenderness"], MODELS[model]["BT"],
                                     SPACING[config], weights)
        common = dict(hull=hull, velocity=v, mass=weight / G, xyz_cg=[x_cb, 0.0, -2.0 * d["T"] / 3.0], water=water,
                      demihull_spacing=spacing, wave_length_scales=hull.default_wave_length_scales(spacing),
                      include_interference_breakdown=False, include_transom_drag=True, hull_dynamics=dynamics)
        row = dict(model=model, config=config, corrected=corrected, Fn=fn)
        try:
            try:
                eq = CraftBuildup.solve_equilibrium(sinkage_guess=guess[0], trim_guess=guess[1],
                                                    max_iterations=80, tolerance=2e-4, **common)
            except RuntimeError:
                eq = CraftBuildup.solve_equilibrium(max_iterations=80, tolerance=2e-4, **common)
            sinkage, trim = float(eq["sinkage"]), float(eq["trim"])
            guess = (sinkage, trim)
            res = CraftBuildup(sinkage=sinkage, trim=trim, **common).run()
            q = 0.5 * rho * v**2 * n_hulls * MODELS[model]["area"]
            row.update(sinkage_pct_draught=100 * sinkage / d["T"], trim_deg=trim,
                       CT_e3=1e3 * float(np.ravel(res["D"])[0]) / q, CW_e3=1e3 * float(np.ravel(res["D_wave"])[0]) / q)
        except Exception as error:
            row["error"] = str(error)[:200]
        rows.append(row)
    return rows


def report():
    rows = json.loads((HERE / "phase_a_results.json").read_text())
    measured = {}
    for name, keys in (("r71_attitude.csv", ("sinkage_pct_draught", "trim_deg")), ("r71_resistance.csv", ("CT_e3", "CWP_e3"))):
        for r in csv.DictReader(open(HERE / name)):
            if not r["flag"]:
                measured.setdefault((r["model"], r["config"], float(r["Fn"])), {}).update({k: float(r[k]) for k in keys})
    bands = [("0.20-0.35", 0.2, 0.35), ("0.40-0.60", 0.4, 0.6), ("0.65-1.00", 0.65, 1.0)]
    groups = [("monohull", lambda r: r["config"] == "mono"), ("catamaran", lambda r: r["config"] != "mono")]

    def errors(key, corrected, pick, lo=0.0, hi=9.0, model=None):
        out = []
        for r in rows:
            m = measured.get((r["model"], r["config"], r["Fn"]), {})
            if (r["corrected"] == corrected and "error" not in r and key in m and pick(r)
                    and lo - 1e-9 <= r["Fn"] <= hi + 1e-9 and (model is None or r["model"] == model)):
                out.append((r[key] - m[key]) if key != "CT_e3" else (r[key] / m[key] - 1) * 100)
        return np.array(out)

    stat = lambda e: f"{np.mean(e):+.2f} / {np.sqrt(np.mean(e**2)):.2f}" if len(e) else "-"
    out = [f"# Phase A: corrected dynamic force and moment, closed loop (closure `{CLOSURE}`)", "",
           "Free running attitude solved with plain thin ship and with the correction fitted without the model being "
           "tested. Each cell: mean error / rms error against Report 71.", ""]
    for key, title in (("trim_deg", "Trim [deg]"), ("sinkage_pct_draught", "Sinkage [% of draught]"), ("CT_e3", "Total resistance [% error]")):
        out += [f"## {title}", "", "| | " + " | ".join(f"Fn {b[0]}" for b in bands) + " | all speeds |", "|---|---|---|---|---|"]
        for label, pick in groups:
            for corrected in (False, True):
                cells = [stat(errors(key, corrected, pick, lo, hi)) for _, lo, hi in bands] + [stat(errors(key, corrected, pick))]
                out.append(f"| {label}, {'corrected' if corrected else 'thin ship'} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["## By model, all configurations and speeds: rms error, thin ship -> corrected", "",
            "| model | trim [deg] | sinkage [% draught] |", "|---|---|---|"]
    rms = lambda e: f"{np.sqrt(np.mean(e**2)):.2f}"
    everything = lambda r: True
    for model in sorted({r["model"] for r in rows}):
        out.append(f"| {model} | {rms(errors('trim_deg', False, everything, model=model))} -> {rms(errors('trim_deg', True, everything, model=model))} "
                   f"| {rms(errors('sinkage_pct_draught', False, everything, model=model))} -> {rms(errors('sinkage_pct_draught', True, everything, model=model))} |")
    failed = [(r["model"], r["config"], r["corrected"], r["Fn"]) for r in rows if "error" in r]
    out += ["", f"Gate (plan): held-out trim within 0.3 deg and sinkage within 1% of draught. "
            f"Corrected, all points: trim rms {rms(errors('trim_deg', True, everything))} deg, "
            f"sinkage rms {rms(errors('sinkage_pct_draught', True, everything))}% of draught.",
            "", f"Solves that found no equilibrium: {len(failed)}" + (f": {failed[:20]}" if failed else "") + "."]
    (HERE / "report").mkdir(exist_ok=True)
    (HERE / "report" / "phase_a.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["fit", "solve", "report"])
    parser.add_argument("--jobs", type=int, default=18)
    args = parser.parse_args()
    if args.step == "fit":
        fit()
    elif args.step == "solve":
        import hydrosandbox as hsb
        from npl import MODELS

        tasks = [(m, c, corrected) for corrected in (True, False) for c in SPACING for m in MODELS]
        rows = [r for part in hsb.parallel_map(solve_curve, tasks, jobs=args.jobs, progress=True) for r in part]
        (HERE / "phase_a_results.json").write_text(json.dumps(rows))
        report()
    else:
        report()
