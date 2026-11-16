"""
Does the Phase A correction, fitted on Report 71's NPL hulls, transfer to a
Series 64 hull (Report 106, model 5s: C_B 0.53 against the NPL series' 0.40)?

    uv run python validation/series64/generalisation.py [--jobs 6]

Open loop: the dynamic force and moment inferred from the measured attitude,
against thin ship and thin ship + correction. Closed loop: the free running
attitude solved both ways, against the measured trim and sinkage. The tow point
is taken as in Report 71 (at the LCG, a third of the draught above the keel);
Report 106 is the same laboratory's but does not restate it. Writes `summary.md`.
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
sys.path.insert(0, str(HERE.parent / "southampton"))
G = 9.80665
SPACING = {"mono": None, "s0.2": 0.2, "s0.4": 0.4}


def curve(config: str) -> list:
    warnings.filterwarnings("ignore")
    import hydrosandbox as hsb
    from hydrosandbox.hull import CraftBuildup
    from phase_a import CorrectedDynamics, features
    from series64 import AREA, BT, series64_hull

    weights = json.loads((HERE.parent / "southampton" / "phase_a_weights.json").read_text())["weights"]["all"]
    water = hsb.Water.freshwater(temperature_C=15.0)
    rho = float(water.density())
    hull, d = series64_hull()
    statics = hull.hydrostatics()
    volume, x_cb = float(statics["volume"]), float(statics["x_cb"])
    slenderness = d["L"] / volume ** (1 / 3)
    hull.transom_closure = 6.0 * float(np.interp(0.0, hull.levels, hull.half_breadths[-1]))
    n_hulls = 1 if SPACING[config] is None else 2
    spacing = None if SPACING[config] is None else SPACING[config] * d["L"]
    weight = n_hulls * volume * rho * G
    rows, guess = [], {False: (0.0, 0.0), True: (0.0, 0.0)}
    for r in csv.DictReader(l for l in open(HERE / "r106_attitude.csv") if not l.startswith("#")):
        fn = float(r["Fn"])
        v = fn * math.sqrt(G * d["L"])
        sink_m, trim_m = float(r[f"{config}_sinkage"]), float(r[f"{config}_trim"])
        common = dict(hull=hull, velocity=v, mass=weight / G, xyz_cg=[x_cb, 0.0, -2.0 * d["T"] / 3.0], water=water,
                      demihull_spacing=spacing, wave_length_scales=hull.default_wave_length_scales(spacing),
                      include_interference_breakdown=False, include_transom_drag=True)
        res = CraftBuildup(sinkage=sink_m / 100 * d["T"], trim=trim_m, **common).run()
        value = lambda key: float(np.ravel(res[key])[0])
        f_thin, m_thin = value("F_z_hull_dynamic") / weight, value("M_y_hull_dynamic") / (weight * d["L"])
        ff, mf = features(fn, slenderness, BT, SPACING[config], f_thin, m_thin)
        row = dict(config=config, Fn=fn, sinkage_measured=sink_m, trim_measured=trim_m,
                   force_measured=f_thin - value("F_z_net") / weight, force_thin=f_thin,
                   force_corrected=f_thin + float(ff @ np.array(weights[0])),
                   moment_measured=m_thin - value("M_y_net") / (weight * d["L"]), moment_thin=m_thin,
                   moment_corrected=m_thin + float(mf @ np.array(weights[1])))
        for corrected in (False, True):
            dynamics = CorrectedDynamics(hull, v, rho, weight, d["L"], slenderness, BT, SPACING[config],
                                         weights if corrected else None)
            tag = "corrected" if corrected else "thin"
            try:
                try:
                    eq = CraftBuildup.solve_equilibrium(sinkage_guess=guess[corrected][0], trim_guess=guess[corrected][1],
                                                        max_iterations=80, tolerance=2e-4, hull_dynamics=dynamics, **common)
                except RuntimeError:
                    eq = CraftBuildup.solve_equilibrium(max_iterations=80, tolerance=2e-4, hull_dynamics=dynamics, **common)
                guess[corrected] = (float(eq["sinkage"]), float(eq["trim"]))
                row[f"sinkage_{tag}"], row[f"trim_{tag}"] = 100 * float(eq["sinkage"]) / d["T"], float(eq["trim"])
            except Exception:
                row[f"sinkage_{tag}"] = row[f"trim_{tag}"] = float("nan")
        rows.append(row)
    return rows


if __name__ == "__main__":
    import hydrosandbox as hsb

    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    rows = [r for part in hsb.parallel_map(curve, list(SPACING), jobs=args.jobs) for r in part]
    (HERE / "results.json").write_text(json.dumps(rows))
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    out = ["# Series 64 model 5s (Report 106): does the Report 71 correction transfer?", "",
           "| configuration | quantity | thin ship rms error | corrected rms error |", "|---|---|---|---|"]
    for config in list(SPACING) + ["all"]:
        sel = [r for r in rows if config in ("all", r["config"])]
        for label, key, scale in (("dynamic force [% of weight]", "force", 100.0), ("trim moment [1e-3 W L]", "moment", 1000.0),
                                  ("trim, closed loop [deg]", "trim", 1.0), ("sinkage, closed loop [% of draught]", "sinkage", 1.0)):
            e = lambda tag: rms([scale * (r[f"{key}_{tag}"] - r[f"{key}_measured"]) for r in sel])
            out.append(f"| {config} | {label} | {e('thin'):.2f} | {e('corrected'):.2f} |")
    out += ["", "## Monohull by speed", "", "| Fn | trim: measured | thin ship | corrected | sinkage: measured | thin ship | corrected |",
            "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["config"] == "mono":
            out.append(f"| {r['Fn']:.2f} | {r['trim_measured']:.2f} | {r['trim_thin']:.2f} | {r['trim_corrected']:.2f} "
                       f"| {r['sinkage_measured']:.1f} | {r['sinkage_thin']:.1f} | {r['sinkage_corrected']:.1f} |")
    (HERE / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))
