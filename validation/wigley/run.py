"""
Michell wave resistance of the Wigley hull against Ju's (1983) fixed-model
measurements: the cleanest test of the wave integral alone (analytic hull, no
transom, no sinkage or trim).

    uv run python validation/wigley/run.py

Writes `summary.md` and `wigley.png` beside this file.
"""

import csv
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from hydrosandbox.hull import Hull
from hydrosandbox.library import kelvin_wave_resistance

HERE = Path(__file__).parent
G, RHO, L = 9.80665, 1000.0, 10.0

rows = list(csv.DictReader(l for l in open(HERE / "ju1983.csv") if not l.startswith("#")))
hull = Hull.wigley(L, L / 10, L / 16)
S = float(hull.hydrostatics()["wetted_surface"])
fr = np.array([float(r["Fr"]) for r in rows])
cw = []
for f in fr:
    u = f * np.sqrt(G * L)
    r_wave = kelvin_wave_resistance(velocity=u, density=RHO, length_scales=(1.0, 15.0, 0.02), **hull.wave_sources(u))
    cw.append(float(np.ravel(r_wave)[0]) / (0.5 * RHO * u**2 * S))
cw = np.array(cw)
cw_paper = np.array([float(r["Cw"]) for r in rows])
wake = np.array([float(r["Ct_minus_Cvw"]) if r["Ct_minus_Cvw"] else np.nan for r in rows])

lines = ["# Wigley hull, held fixed (Ju 1983): Michell C_W against measurement", "",
         "| Fr | measured C_w x 1000 | C_t - C_vw x 1000 | Michell C_W x 1000 | Michell / measured |", "|---|---|---|---|---|"]
for i in range(0, len(fr), 2):
    if fr[i] >= 0.16:
        lines.append(f"| {fr[i]:.2f} | {1e3 * cw_paper[i]:.2f} | {1e3 * wake[i]:.2f} | {1e3 * cw[i]:.2f} | {cw[i] / cw_paper[i]:.2f} |")
for lo, hi in ((0.16, 0.25), (0.25, 0.32), (0.32, 0.40)):
    m = (fr >= lo) & (fr <= hi)
    lines.append(f"\nFr {lo}-{hi}: median Michell / measured {np.median(cw[m] / cw_paper[m]):.2f}, "
                 f"rms difference {1e3 * np.sqrt(np.mean((cw[m] - cw_paper[m]) ** 2)):.2f} x 1e-3")
(HERE / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
ax.plot(fr, 1e3 * cw_paper, color="#0b0b0b", marker="o", markersize=4, linestyle="none", label="measured C_w (Ju 1983)")
ax.plot(fr, 1e3 * wake, color="#898781", marker="s", markersize=4, linestyle="none", label="measured C_t - C_vw (wake survey)")
ax.plot(fr, 1e3 * cw, color="#2a78d6", linewidth=2, label="HydroSandbox (Michell)")
ax.set_xlabel("Froude number"); ax.set_ylabel("wave resistance coefficient x 1000")
ax.set_title("Wigley hull held at zero sinkage and trim", loc="left", fontsize=12)
ax.grid(True, color="#e1e0d9", linewidth=0.6); ax.legend(frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout(); fig.savefig(HERE / "wigley.png", dpi=110)
