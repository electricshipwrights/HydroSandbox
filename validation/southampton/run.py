"""
Run HydroSandbox's hull model on the Southampton catamaran series and store what
it predicts where Report 71 has measurements.

    uv run python validation/southampton/run.py [--models 5b,6b] [--configs mono,s0.3]
                                                [--closures auto,none] [--jobs 6]

One task is one resistance curve (a model, a configuration and a transom
closure, Fn 0.2 to 1.0), solved for free sinkage and trim at each speed and
written to `results/<model>_<config>_<closure>.json`. Curves already there are
skipped, so a run can be stopped and resumed. Then: `report.py`.

Test conditions follow the report: 1.6 m models in fresh water (rho 1000,
nu 1.141e-6), free to sink and trim, towed horizontally from the LCG at a third
of the draught above the keel; coefficients on the static wetted area of
Table II. Closures: "auto" and "none" as in `Hull`, a number for a fixed hollow
length [m], or "f0.2" for a hollow of that fraction of the transverse wavelength.
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULTS = HERE / "results"
G = 9.80665
CONFIGS = {"mono": None, "s0.2": 0.2, "s0.3": 0.3, "s0.4": 0.4, "s0.5": 0.5}
FROUDE = [round(0.2 + 0.05 * i, 2) for i in range(17)]


def closure_value(closure: str, froude: float, length: float):
    if closure in ("auto", "boatmath", "none"):
        return closure
    if closure.startswith("f"):
        return float(closure[1:]) * 2 * math.pi * froude**2 * length
    return float(closure)


def run_curve(model: str, config: str, closure: str) -> str:
    warnings.filterwarnings("ignore")
    import hydrosandbox as hsb
    from hydrosandbox.hull import CraftBuildup
    from npl import MODELS, npl_hull

    water = hsb.Water.freshwater(temperature_C=15.0)
    rho = float(water.density())
    hull, d = npl_hull(model)
    statics = hull.hydrostatics()
    volume, x_cb = float(statics["volume"]), float(statics["x_cb"])
    n_hulls = 1 if CONFIGS[config] is None else 2
    spacing = None if CONFIGS[config] is None else CONFIGS[config] * d["L"]
    rows, guess = [], (0.0, 0.0)
    for fn in FROUDE:
        hull.transom_closure = closure_value(closure, fn, d["L"])
        v = fn * math.sqrt(G * d["L"])
        common = dict(
            hull=hull, velocity=v, mass=n_hulls * volume * rho,
            xyz_cg=[x_cb, 0.0, -2.0 * d["T"] / 3.0], water=water,
            demihull_spacing=spacing,
            wave_length_scales=hull.default_wave_length_scales(spacing),
            include_interference_breakdown=False, include_transom_drag=True,
        )
        row = dict(model=model, config=config, closure=closure, Fn=fn)
        try:
            try:
                eq = CraftBuildup.solve_equilibrium(
                    sinkage_guess=guess[0], trim_guess=guess[1],
                    max_iterations=80, tolerance=2e-4, **common,
                )
            except RuntimeError:  # a poor warm start: try from rest
                eq = CraftBuildup.solve_equilibrium(max_iterations=80, tolerance=2e-4, **common)
            sinkage, trim = float(eq["sinkage"]), float(eq["trim"])
            guess = (sinkage, trim)
            res = CraftBuildup(sinkage=sinkage, trim=trim, **common).run()
            q = 0.5 * rho * v**2 * n_hulls * MODELS[model]["area"]
            value = lambda key: float(np.ravel(res[key])[0])
            row.update(
                CW_e3=1e3 * value("D_wave") / q,
                CV_e3=1e3 * value("D_hull_viscous") / q,
                Ctransom_e3=1e3 * value("D_transom") / q,
                CT_e3=1e3 * value("D") / q,
                sinkage_pct_draught=100 * sinkage / d["T"],
                trim_deg=trim,
                reynolds=v * d["L"] / float(water.kinematic_viscosity()),
            )
        except Exception as error:  # keep the curve; record the hole
            row["error"] = str(error)[:200]
        rows.append(row)
    path = RESULTS / f"{model}_{config}_{closure}.json"
    path.write_text(json.dumps(rows, indent=1))
    return path.name


if __name__ == "__main__":
    import hydrosandbox as hsb
    from npl import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--closures", default="auto,none")
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    tasks = [
        (m, c, cl)
        for cl in args.closures.split(",")
        for c in args.configs.split(",")
        for m in args.models.split(",")
        if not (RESULTS / f"{m}_{c}_{cl}.json").exists()
    ]
    print(f"{len(tasks)} curves to run", flush=True)
    hsb.parallel_map(run_curve, tasks, jobs=args.jobs, progress=True, labels=["_".join(t) for t in tasks])
