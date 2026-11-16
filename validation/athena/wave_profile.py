"""
The wave profile along the hull: thin ship against Jenkins' (1984) measurements
on R/V Athena held at zero sinkage and trim.

    uv run python validation/athena/wave_profile.py

Elevations are given as zeta / (U^2 / 2g), positive UP (the report's sign is
reversed). The thin-ship elevation is the linear wave field evaluated on the
still waterline at the hull side. Writes `wave_profile.md` and `.png`.
"""

import csv
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from athena import LPP, athena_hull

HERE = Path(__file__).parent
G = 9.80665
rows = list(csv.DictReader(l for l in open(HERE / "jenkins1984_wave_profile_fixed.csv") if not l.startswith("#")))
stations = np.array([float(r["station"]) for r in rows])
froudes = [0.28, 0.35, 0.41, 0.48, 0.65]
measured = {fn: -np.array([float(r[f"Fn{fn:.2f}"]) for r in rows]) for fn in froudes}

hull = athena_hull(transom_closure="none")
x = stations / 20.0 * LPP
y = np.array([np.interp(0.0, hull.levels, hull.half_breadths[np.argmin(np.abs(hull.stations - xi))]) for xi in x])
model = {}
for fn in froudes:
    v = fn * math.sqrt(G * LPP)
    field = hull.wave_field(v, G, level=2)
    model[fn] = np.ravel(field.elevation(x, y)) / (v**2 / (2 * G))

out = ["# Wave profile along R/V Athena's hull, held fixed: thin ship against Jenkins (1984)", "",
       "zeta / (U^2 / 2g), positive up. Bow crest = the highest point over stations 0-6.", "",
       "| Fn | measured bow crest (station) | thin-ship bow crest (station) | ratio | mean over stations 0-4: measured, thin ship | U^2/2g [mm] |",
       "|---|---|---|---|---|---|"]
fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=True, facecolor="#fcfcfb")
for ax, fn in zip(axes, froudes):
    fore = stations <= 6
    im, ic = int(np.argmax(measured[fn][fore])), int(np.argmax(model[fn][fore]))
    head = 1000 * (fn * math.sqrt(G * LPP)) ** 2 / (2 * G)
    out.append(f"| {fn:.2f} | {measured[fn][im]:.3f} ({stations[im]:.0f}) | {model[fn][ic]:.3f} ({stations[ic]:.0f}) | "
               f"{model[fn][ic] / measured[fn][im]:.2f} | {measured[fn][:5].mean():.3f}, {model[fn][:5].mean():.3f} | {head:.0f} |")
    ax.plot(stations, measured[fn], color="#0b0b0b", marker="o", markersize=4, linestyle="none", label="measured (Jenkins 1984)")
    ax.plot(stations, model[fn], color="#2a78d6", linewidth=2, label="thin ship")
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_title(f"Fn {fn:.2f}", loc="left", fontsize=10); ax.set_xlabel("station (0 = bow)", fontsize=9)
    ax.set_facecolor("#fcfcfb"); ax.grid(True, color="#e1e0d9", linewidth=0.6); ax.tick_params(colors="#898781", labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("elevation / (U²/2g), up", fontsize=9); axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("Wave profile along the hull, R/V Athena held fixed", x=0.01, ha="left", fontsize=12)
fig.tight_layout(); fig.savefig(HERE / "wave_profile.png", dpi=110)
out += ["", "| station | " + " | ".join(f"Fn {fn:.2f}: meas, thin" for fn in froudes) + " |", "|---|" + "---|" * len(froudes)]
for i, st in enumerate(stations):
    out.append(f"| {st:.0f} | " + " | ".join(f"{measured[fn][i]:+.3f}, {model[fn][i]:+.3f}" for fn in froudes) + " |")
(HERE / "wave_profile.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
