"""
Do two hulls' measured wave spectra add? Insel's (1990) Fig. 159: the Wigley
hull held fixed at Fn 0.50, monohull and catamarans at four spacings, resolved
by tank harmonic.

    uv run python validation/wigley/insel_fig159.py

For each catamaran harmonic: the measured value; the measured *monohull*
harmonic times the linear placement factor 4 cos^2(ky S / 2) (superposition
with nothing else); and thin ship. All per demihull, in the figure's units.
See `insel1990_fig159.csv` for how the figure was read and its normalisation.
Writes `insel_fig159.md`.
"""

import csv
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

warnings.filterwarnings("ignore")
from hydrosandbox.hull import Hull

HERE = Path(__file__).parent
G, RHO, L, TANK_WIDTH, FN = 9.80665, 1000.0, 1.8, 3.70, 0.5

rows = list(csv.DictReader(l for l in open(HERE / "insel1990_fig159.csv") if not l.startswith("#")))
data = {(r["config"], int(r["n"])): r for r in rows}

hull = Hull.wigley(L, L / 10, L / 16, n_stations=121, n_levels=61)
v = FN * np.sqrt(G * L)
nu = G / v**2
ky = 2 * np.pi * np.arange(3) / TANK_WIDTH
sigma = np.array([0.0] + [brentq(lambda s: s * np.sqrt(nu + s * s) - k, 0, 1e3) for k in ky[1:]])
kappa = nu + sigma**2
re, im = (np.ravel(np.asarray(a, float)) for a in hull.wave_amplitude(v, G)(np.sqrt(nu * kappa), kappa, ky))
weight = np.full(3, 2 * np.pi / TANK_WIDTH)
weight[0] *= 0.5  # the n = 0 mode counts once, the others stand for +/- ky
thin = (RHO * G**2 / (np.pi * v**4 * np.sqrt(nu)) * (re**2 + im**2) * np.sqrt(kappa / nu)
        * np.sqrt(kappa) / (nu + 2 * sigma**2) * weight) / v**2 / 2  # monohull, figure units

lines = ["# Wigley hull held fixed, Fn 0.50 (Insel 1990, Fig. 159): do the hulls' spectra add?", "",
         "Per demihull, in the figure's units; readings are +/- 0.01.", "",
         "Monohull, by harmonic (0, 32.7, 46.5 deg): measured "
         + ", ".join(data["mono", n]["measured"] for n in range(3))
         + "; thin ship " + ", ".join(f"{t:.3f}" for t in thin)
         + "; thesis's theory " + ", ".join(data["mono", n]["thesis_theory"] for n in range(3)) + ".", "",
         "| S/L | harmonic | measured catamaran | measured monohull x placement factor | thin ship | thesis's own prediction |",
         "|---|---|---|---|---|---|"]
errors = {"superposed": [], "thin": [], "thesis": []}
for config in ("s0.2", "s0.3", "s0.4", "s0.5"):
    spacing = float(config[1:]) * L
    for n in range(3):
        pair = 4 * np.cos(0.5 * ky[n] * spacing) ** 2 / 2  # per demihull
        meas = float(data[config, n]["measured"])
        superposed = pair * float(data["mono", n]["measured"])
        thesis = float(data[config, n]["thesis_predicted"])
        errors["superposed"].append(superposed - meas); errors["thin"].append(pair * thin[n] - meas)
        errors["thesis"].append(thesis - meas)
        lines.append(f"| {config[1:]} | {n} | {meas:.3f} | {superposed:.3f} | {pair * thin[n]:.3f} | {thesis:.3f} |")
rms = {k: np.sqrt(np.mean(np.square(e))) for k, e in errors.items()}
lines += ["", f"rms difference from the measured catamaran harmonics: superposed measured monohull {rms['superposed']:.3f}, "
          f"thin ship {rms['thin']:.3f}, thesis's prediction {rms['thesis']:.3f}."]
(HERE / "insel_fig159.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
