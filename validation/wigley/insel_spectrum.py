"""
Where in the wave spectrum thin ship is wrong: Insel's (1990) measured
wave-pattern spectrum of a Wigley hull at Fn 0.331 against the Michell spectrum,
by wave direction theta.

    uv run python validation/wigley/insel_spectrum.py

The measurement is a discrete tank spectrum (one harmonic per transverse tank
mode); the Michell spectrum is integrated over the same theta intervals, each
running between the midpoints of neighbouring harmonics. The hull is taken at
zero sinkage and trim. Writes `insel_spectrum.md`.
"""

import csv
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
from hydrosandbox.hull import Hull
from hydrosandbox.library import wave_quadrature_nodes

HERE = Path(__file__).parent
G, RHO, L, FN, S_REPORT = 9.80665, 1000.0, 1.8, 0.33097, 0.482

rows = list(csv.DictReader(l for l in open(HERE / "insel1990_spectrum.csv") if not l.startswith("#")))
theta = np.array([float(r["theta_deg"]) for r in rows])
measured = np.array([float(r["Rwp_N"]) for r in rows])
wn = np.array([float(r["wn"]) for r in rows])

hull = Hull.wigley(L, L / 10, L / 16)
v = FN * np.sqrt(G * L)
nu = G / v**2
sigma, weights = wave_quadrature_nodes(nu, 1.0, 1.5 * L, 0.004)
kappa = nu + sigma**2
re, im = (np.ravel(np.asarray(a, dtype=float)) for a in hull.wave_amplitude(v, G)(np.sqrt(nu * kappa), kappa, sigma * np.sqrt(kappa)))
density = RHO * G**2 / (np.pi * v**4 * np.sqrt(nu)) * (re**2 + im**2) * np.sqrt(kappa / nu) * weights  # [N] per node
node_theta = np.degrees(np.arctan(sigma / np.sqrt(nu)))
edges = np.concatenate([[0.0], 0.5 * (theta[1:] + theta[:-1]), [90.0]])
michell = np.array([density[(node_theta >= lo) & (node_theta < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
beyond = michell[-1] - density[(node_theta >= edges[-2]) & (node_theta < theta[-1] + 0.2)].sum()

q = 0.5 * RHO * v**2 * S_REPORT
lines = ["# Wigley hull, Fn 0.331: measured wave-pattern spectrum (Insel 1990, Table 3) against Michell", "",
         f"Measured R_WP {measured.sum():.3f} N (C_WP x 1000 = {1e3 * measured.sum() / q:.2f}); "
         f"Michell {michell.sum():.3f} N (C_W x 1000 = {1e3 * michell.sum() / q:.2f}), of which {beyond:.3f} N lies "
         f"beyond the last measured harmonic (theta > {theta[-1]:.1f} deg).", "",
         f"Check: the tabulated wavenumbers are the deep-water longitudinal ones, wn = nu / cos(theta), "
         f"to within {100 * np.max(np.abs(wn * np.cos(np.radians(theta)) / nu - 1)):.2f}%.", "",
         "| theta band [deg] | measured [N] | Michell [N] | Michell / measured |", "|---|---|---|---|"]
for lo, hi in ((0, 25), (25, 35), (35, 41), (41, 46), (46, 53), (53, 61), (61, 66), (66, 90)):
    m = (theta >= lo) & (theta < hi)
    lines.append(f"| {lo}-{hi} | {measured[m].sum():.3f} | {michell[m].sum():.3f} | {michell[m].sum() / measured[m].sum():.2f} |")
(HERE / "insel_spectrum.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
