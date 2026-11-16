"""
HydroSandbox's hull model against the R/V Athena model tests (Jenkins 1984),
held at zero sinkage and trim and free to sink and trim.

    uv run python validation/athena/run.py [--closures auto,none] [--jobs 4]

Writes `results.json`, `summary.md` and `athena.png` beside this file. The fixed
runs test the wave integral on its own; the free runs add the dynamic sinkage
force and trim moment. The report does not give the tow point, so the tow force
is taken through the centre of gravity (no moment), with the centre of gravity
over the centre of buoyancy; fresh water at 15 C.
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


def measured():
    rows = [r for r in csv.DictReader(l for l in open(HERE / "jenkins1984.csv") if not l.startswith("#"))]
    return [{k: (float(v) if v else None) for k, v in r.items()} for r in rows]


def run_curve(condition: str, closure: str) -> list:
    warnings.filterwarnings("ignore")
    import hydrosandbox as hsb
    from hydrosandbox.hull import CraftBuildup
    from athena import LPP, WETTED_SURFACE, athena_hull

    water = hsb.Water.freshwater(temperature_C=15.0)
    rho = float(water.density())
    hull = athena_hull(transom_closure=closure)
    statics = hull.hydrostatics()
    fns = sorted({r["Fn"] for r in measured()} | {r[f"Fn_wp_{condition}"] for r in measured() if r[f"Fn_wp_{condition}"]})
    rows, guess = [], (0.0, 0.0)
    for fn in fns:
        v = fn * math.sqrt(G * LPP)
        common = dict(
            hull=hull, velocity=v, mass=float(statics["volume"]) * rho,
            xyz_cg=[float(statics["x_cb"]), 0.0, 0.0], water=water,
            wave_length_scales=hull.default_wave_length_scales(None),
            include_interference_breakdown=False, include_transom_drag=True,
        )
        row = dict(condition=condition, closure=closure, Fn=fn)
        try:
            sinkage = trim = 0.0
            if condition == "free":
                try:
                    eq = CraftBuildup.solve_equilibrium(sinkage_guess=guess[0], trim_guess=guess[1],
                                                        max_iterations=80, tolerance=2e-4, **common)
                except RuntimeError:
                    eq = CraftBuildup.solve_equilibrium(max_iterations=80, tolerance=2e-4, **common)
                sinkage, trim = float(eq["sinkage"]), float(eq["trim"])
                guess = (sinkage, trim)
            res = CraftBuildup(sinkage=sinkage, trim=trim, **common).run()
            q = 0.5 * rho * v**2 * WETTED_SURFACE
            value = lambda key: float(np.ravel(res[key])[0])
            # Rise of midships / Lpp, as the report's Cs (the pivot is the centre of gravity).
            rise_mid = -sinkage + (float(statics["x_cb"]) - 0.5 * LPP) * math.tan(math.radians(trim))
            row.update(CW_e3=1e3 * value("D_wave") / q, CV_e3=1e3 * value("D_hull_viscous") / q,
                       Ctransom_e3=1e3 * value("D_transom") / q, CT_e3=1e3 * value("D") / q,
                       Cs_e2=100 * rise_mid / LPP, Ctau_e2=100 * math.tan(math.radians(trim)), trim_deg=trim)
        except Exception as error:
            row["error"] = str(error)[:200]
        rows.append(row)
    return rows


def report(results, closures):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = measured()
    ink, muted, grid, surface = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    colour = dict(zip(closures, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
    curve = lambda cond, cl: {r["Fn"]: r for r in results if r["condition"] == cond and r["closure"] == cl and "error" not in r}
    panels = [
        ("Wave pattern, held fixed", "C_W x 1000", "fixed", "Fn_wp_fixed", "CWP_fixed_e3", "CW_e3"),
        ("Wave pattern, free", "C_W x 1000", "free", "Fn_wp_free", "CWP_free_e3", "CW_e3"),
        ("Total resistance, held fixed", "C_T x 1000", "fixed", "Fn", "CT_fixed_e3", "CT_e3"),
        ("Total resistance, free", "C_T x 1000", "free", "Fn", "CT_free_e3", "CT_e3"),
        ("Trim, free", "(bow rise - stern rise) / L x 100", "free", "Fn", "Ctau_e2", "Ctau_e2"),
        ("Midship rise, free", "rise / L x 100", "free", "Fn", "Cs_e2", "Cs_e2"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), facecolor=surface, sharex=True)
    lines = ["# R/V Athena (Jenkins 1984): HydroSandbox against measurement", ""]
    for ax, (title, ylabel, cond, fn_key, m_key, c_key) in zip(axes.ravel(), panels):
        pts = [(r[fn_key], r[m_key]) for r in data if r[fn_key] is not None and r[m_key] is not None]
        ax.plot(*zip(*pts), color=ink, marker="o", markersize=4, linestyle="none", label="measured (Jenkins 1984)")
        lines += [f"## {title}: {ylabel}", "", "| Fn | measured | " + " | ".join(f"model, {c}" for c in closures) + " |",
                  "|" + "---|" * (2 + len(closures))]
        for closure in closures:
            c = curve(cond, closure)
            ax.plot(sorted(c), [c[f][c_key] for f in sorted(c)], color=colour[closure], linewidth=2,
                    label=f"model, transom closure '{closure}'")
        for fn, m in pts:
            cells = [f"{curve(cond, cl)[fn][c_key]:.2f}" if fn in curve(cond, cl) else "-" for cl in closures]
            lines.append(f"| {fn:.3f} | {m:.2f} | " + " | ".join(cells) + " |")
        lines.append("")
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.set_ylabel(ylabel, fontsize=9, color=ink)
        ax.set_facecolor(surface)
        ax.grid(True, color=grid, linewidth=0.6)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if c_key in ("CW_e3", "CT_e3"):
            ax.set_ylim(0, 4 if c_key == "CW_e3" else 9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[2]:
        ax.set_xlabel("Froude number", fontsize=9, color=ink)
    fig.suptitle("R/V Athena model 5365 (L/B 7.5, B/T 4.2, transom stern)", x=0.02, ha="left", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(HERE / "athena.png", dpi=110)
    failures = [(r["condition"], r["closure"], r["Fn"]) for r in results if "error" in r]
    lines.append(f"Solves that failed: {len(failures)} {failures if failures else ''}")
    (HERE / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import hydrosandbox as hsb

    parser = argparse.ArgumentParser()
    parser.add_argument("--closures", default="auto,none")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    closures = args.closures.split(",")
    path = HERE / "results.json"
    if not args.report_only:
        tasks = [(cond, cl) for cond in ("fixed", "free") for cl in closures]
        results = [r for rows in hsb.parallel_map(run_curve, tasks, jobs=args.jobs, progress=True) for r in rows]
        path.write_text(json.dumps(results, indent=1))
    report(json.loads(path.read_text()), closures)
