"""
Thin-ship wave spectra of the Southampton models at the *measured* running
attitude, for every model, configuration, speed and transom closure.

    uv run python validation/southampton/spectra.py [--closures auto,none,c6,couser] [--jobs 4]

For each case the wave resistance is split by wave direction theta (the angle
between a wave component's travel and the hull's track): `energy[k]` is the
part of `C_W x 1000` carried by waves with theta in the k-th one-degree bin, so
`energy.sum()` is the thin-ship `C_W x 1000`. A catamaran's spectrum is the
demihull's times the placement factor `4 cos^2(ky S / 2)`. Using the measured
sinkage and trim takes the model's own attitude error out of the comparison.
Writes `spectra.npz`; `spectrum_gain.py` reads it.
"""

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
G, RHO = 9.80665, 1000.0
SPACING = {"mono": None, "s0.2": 0.2, "s0.3": 0.3, "s0.4": 0.4, "s0.5": 0.5}
N_BINS = 90


def binned_spectrum(hull, d, area, fn, sinkage, trim, spacings, x_cb):
    """C_W x 1000 by one-degree theta bin, for each spacing in `spacings` (None = monohull)."""
    from hydrosandbox.library import wave_quadrature_nodes

    v = fn * math.sqrt(G * d["L"])
    nu = G / v**2
    y_extent, x_extent, min_depth = hull.default_wave_length_scales(0.5 * d["L"])
    sigma, weights = wave_quadrature_nodes(nu, y_extent, x_extent, min_depth)
    kappa = nu + sigma**2
    lam = np.sqrt(kappa / nu)
    kx, ky = np.sqrt(nu * kappa), sigma * np.sqrt(kappa)
    amplitude = hull.wave_amplitude(v, G, sinkage, trim, (x_cb, -2.0 * d["T"] / 3.0))
    re, im = (np.ravel(np.asarray(a, dtype=float)) for a in amplitude(kx, kappa, ky))
    density = RHO * G**2 / (math.pi * v**4 * math.sqrt(nu)) * (re**2 + im**2) * lam * weights  # [N] per node
    theta_bin = np.minimum((np.degrees(np.arctan(sigma / math.sqrt(nu)))).astype(int), N_BINS - 1)
    out = {}
    for spacing in spacings:
        n_hulls = 1 if spacing is None else 2
        pair = 1.0 if spacing is None else 4.0 * np.cos(0.5 * spacing * d["L"] * ky) ** 2
        coefficient = 1e3 * density * pair / (0.5 * RHO * v**2 * n_hulls * area)
        out[spacing] = np.bincount(theta_bin, weights=coefficient, minlength=N_BINS)
    return out


# Couser (1996), Fig. 5.8: virtual-appendage (re-attachment) length / transom half-breadth against Fn, by B/T.
COUSER_FN = [0.0, 0.25, 0.55, 0.58, 0.65, 0.70, 1.0]
COUSER_RATIO = {1.5: [8.0, 6.0, 4.0, 3.6, 6.05, 8.0, 8.0], 2.0: [8.0, 5.9, 3.8, 3.4, 5.95, 8.0, 8.0],
                2.5: [8.0, 5.7, 3.4, 2.95, 5.7, 8.0, 8.0]}


def closure_value(closure: str, fn: float, bt: float, transom_half_beam: float):
    """`auto` / `none` as in `Hull`; `c6`: a hollow six transom half-breadths long (Couser's constant
    virtual appendage); `couser`: his variation of that length with Fn and B/T."""
    if closure in ("auto", "boatmath", "none"):
        return closure
    if closure == "couser":
        return float(np.interp(fn, COUSER_FN, COUSER_RATIO[bt])) * transom_half_beam
    return float(closure[1:]) * transom_half_beam


def model_spectra(model: str, closure: str) -> list:
    warnings.filterwarnings("ignore")
    from npl import MODELS, npl_hull

    hull, d = npl_hull(model)
    transom_half_beam = float(np.interp(0.0, hull.levels, hull.half_breadths[-1]))
    x_cb = float(hull.hydrostatics()["x_cb"])
    attitude = {(r["config"], float(r["Fn"])): r for r in csv.DictReader(open(HERE / "r71_attitude.csv"))
                if r["model"] == model and not r["flag"]}
    rows = []
    for r in csv.DictReader(open(HERE / "r71_resistance.csv")):
        key = (r["config"], float(r["Fn"]))
        if r["model"] != model or r["flag"] or key not in attitude:
            continue
        a = attitude[key]
        sinkage = float(a["sinkage_pct_draught"]) / 100 * d["T"]
        spacing = SPACING[r["config"]]
        hull.transom_closure = closure_value(closure, key[1], MODELS[model]["BT"], transom_half_beam)
        energy = binned_spectrum(hull, d, MODELS[model]["area"], key[1], sinkage, float(a["trim_deg"]), [spacing], x_cb)
        rows.append((model, r["config"], closure, key[1], float(r["CWP_e3"]), float(r["CT_e3"]), energy[spacing]))
    return rows


if __name__ == "__main__":
    import hydrosandbox as hsb
    from npl import MODELS

    parser = argparse.ArgumentParser()
    parser.add_argument("--closures", default="auto,none")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    path = HERE / "spectra.npz"
    old = dict(np.load(path)) if path.exists() else None
    have = set(old["closure"]) if old else set()
    # "label=closure" stores a closure under another name (e.g. `pr3=auto` when run from another checkout).
    wanted = [(c.split("=")[0], c.split("=")[-1]) for c in args.closures.split(",")]
    tasks = [(m, closure) for label, closure in wanted if label not in have for m in MODELS]
    labels = [label for label, closure in wanted if label not in have for m in MODELS]
    parts = hsb.parallel_map(model_spectra, tasks, jobs=args.jobs, progress=True)
    rows = [r[:2] + (label,) + r[3:] for part, label in zip(parts, labels) for r in part]
    if old:  # keep the closures already computed
        rows = list(zip(*(old[k] for k in ("model", "config", "closure", "Fn", "CWP_e3", "CT_e3", "energy")))) + rows
    np.savez_compressed(
        path,
        model=np.array([r[0] for r in rows]), config=np.array([r[1] for r in rows]),
        closure=np.array([r[2] for r in rows]), Fn=np.array([r[3] for r in rows]),
        CWP_e3=np.array([r[4] for r in rows]), CT_e3=np.array([r[5] for r in rows]),
        energy=np.array([r[6] for r in rows]),
    )
    print(len(rows), "spectra")
