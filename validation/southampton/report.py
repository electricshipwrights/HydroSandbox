"""
Compare `run.py`'s results with Report 71 and write the error map.

    uv run python validation/southampton/report.py

Writes `report/summary.md` (tables) and `report/*.png` (one small-multiple
figure per quantity, laid out as the series is: length-displacement ratio down,
B/T across). Curves that have not been run yet are left out.

What is compared, per model, configuration and Froude number:

- `C_W` against the measured wave-pattern coefficient `C_WP`: the far-field
  wave energy, which is what the model's wave spectrum is supposed to carry.
- the wave interference factor `tau = C_WP(catamaran) / C_WP(monohull)`,
  measured and modelled: whether two demihulls' waves add as linear theory says.
- `C_T` against `C_T`, and the non-wave remainder `C_T - C_WP` against the
  model's friction + transom drag (and against the report's `(1 + k) C_F`).
- running trim and sinkage.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "report"
CONFIGS = ["mono", "s0.2", "s0.3", "s0.4", "s0.5"]
GRID = [["", "3b", ""], ["4a", "4b", "4c"], ["5a", "5b", "5c"], ["6a", "6b", "6c"]]
MODELS = [m for row in GRID for m in row if m]
BANDS = [("0.20-0.35", 0.2, 0.35), ("0.40-0.60", 0.4, 0.6), ("0.65-1.00", 0.65, 1.0)]
# Report 71, Table III: form factors from the wave-pattern measurements,
# (1 + k) for the monohull and (1 + beta k) at S/L = 0.2, 0.3, 0.4, 0.5.
FORM_FACTORS = {
    "3b": [1.45, 1.60, 1.65, 1.55, 1.60], "4a": [1.30, 1.43, 1.43, 1.46, 1.44],
    "4b": [1.30, 1.47, 1.43, 1.45, 1.45], "4c": [1.30, 1.41, 1.39, 1.48, 1.44],
    "5a": [1.28, 1.44, 1.43, 1.44, 1.47], "5b": [1.26, 1.41, 1.45, 1.40, 1.38],
    "5c": [1.26, 1.41, 1.43, 1.42, 1.44], "6a": [1.22, 1.48, 1.44, 1.46, 1.48],
    "6b": [1.22, 1.42, 1.40, 1.47, 1.44], "6c": [1.23, 1.40, 1.40, 1.45, 1.44],
}
INK, MUTED, GRIDLINE, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = {"auto": "#2a78d6", "none": "#eb6834"}  # further closures take the next slots
EXTRA = ["#1baf7a", "#eda100", "#e87ba4"]


def ittc57(reynolds):
    return 0.075 / (math.log10(reynolds) - 2.0) ** 2


def load():
    measured = defaultdict(dict)  # (model, config) -> Fn -> dict
    for name, keys in (("r71_resistance.csv", ("CT_e3", "CWP_e3")),
                       ("r71_attitude.csv", ("sinkage_pct_draught", "trim_deg"))):
        for row in csv.DictReader(open(HERE / name)):
            if row["flag"]:
                continue  # the report repeats the monohull's entry: not a measurement
            point = measured[(row["model"], row["config"])].setdefault(float(row["Fn"]), {})
            for key in keys:
                point[key] = float(row[key])
    computed = defaultdict(dict)  # (model, config, closure) -> Fn -> dict
    for path in sorted((HERE / "results").glob("*.json")):
        for row in json.loads(path.read_text()):
            if "error" not in row:
                computed[(row["model"], row["config"], row["closure"])][row["Fn"]] = row
    return measured, computed


def pairs(measured, computed, model, config, closure, m_key, c_key, lo=0.0, hi=9.0):
    """(Fn, measured, computed) where both exist."""
    m, c = measured.get((model, config), {}), computed.get((model, config, closure), {})
    return [(fn, m[fn][m_key], c[fn][c_key]) for fn in sorted(m)
            if lo <= fn <= hi and m_key in m[fn] and fn in c]


def tau(curves, model, config, key, fn):
    """Catamaran over monohull wave coefficient."""
    cat, mono = curves(model, config).get(fn), curves(model, "mono").get(fn)
    if not cat or not mono or key not in cat or key not in mono or mono[key] <= 0:
        return None
    return cat[key] / mono[key]


def small_multiples(name, title, ylabel, draw, legend):
    fig, axes = plt.subplots(4, 3, figsize=(12, 12.5), sharex=True, facecolor=SURFACE)
    for i, row in enumerate(GRID):
        for j, model in enumerate(row):
            ax = axes[i, j]
            if not model:
                ax.axis("off")
                continue
            ax.set_facecolor(SURFACE)
            draw(ax, model)
            ax.set_title(f"Model {model}", loc="left", fontsize=10, color=INK)
            ax.grid(True, color=GRIDLINE, linewidth=0.6)
            ax.tick_params(colors=MUTED, labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#c3c2b7")
            if i == 3:
                ax.set_xlabel("Froude number", color=INK, fontsize=9)
            if j == 0 or (i == 0 and j == 1):
                ax.set_ylabel(ylabel, color=INK, fontsize=9)
    handles = [plt.Line2D([], [], **style) for _, style in legend]
    axes[0, 2].legend(handles, [label for label, _ in legend], loc="center", frameon=False, fontsize=10)
    axes[0, 0].text(0.0, 0.55, title, fontsize=13, color=INK, wrap=True, va="center")
    axes[0, 0].text(0.0, 0.15, "Rows: length-displacement ratio 6.3, 7.4, 8.5, 9.5\n"
                    "Columns: B/T 1.5, 2.0, 2.5", fontsize=9, color=MUTED, va="center")
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    measured, computed = load()
    closures = sorted({k[2] for k in computed}, key=lambda c: (c not in SERIES, c))
    colour = dict(SERIES, **{c: EXTRA[i % 3] for i, c in enumerate(c for c in closures if c not in SERIES)})
    dots = dict(color=INK, marker="o", markersize=4, linestyle="none")
    line = lambda c: dict(color=colour[c], linewidth=2)
    c_curves = lambda closure: (lambda model, config: computed.get((model, config, closure), {}))
    m_curves = lambda model, config: measured.get((model, config), {})

    def figure(name, title, ylabel, config, m_key, c_key, ymax=None):
        def draw(ax, model):
            m = measured.get((model, config), {})
            fns = [fn for fn in sorted(m) if m_key in m[fn]]
            ax.plot(fns, [m[fn][m_key] for fn in fns], **dots)
            for closure in closures:
                c = computed.get((model, config, closure), {})
                ax.plot(sorted(c), [c[fn][c_key] for fn in sorted(c)], **line(closure))
            if ymax:
                ax.set_ylim(0, ymax)
        legend = [("measured (Report 71)", dots)] + [(f"model, transom closure '{c}'", line(c)) for c in closures]
        small_multiples(name, title, ylabel, draw, legend)

    figure("cw_mono.png", "Wave resistance, monohull:\nmodel C_W against measured\nwave-pattern C_WP",
           "coefficient x 1000", "mono", "CWP_e3", "CW_e3", ymax=8)
    figure("ct_mono.png", "Total resistance, monohull", "C_T x 1000", "mono", "CT_e3", "CT_e3", ymax=14)
    figure("ct_s0.3.png", "Total resistance, catamaran\nS/L = 0.3", "C_T x 1000", "s0.3", "CT_e3", "CT_e3", ymax=18)
    figure("trim_mono.png", "Running trim, monohull", "trim [deg, bow up]", "mono", "trim_deg", "trim_deg")
    figure("sinkage_mono.png", "Running sinkage, monohull", "sinkage [% of draught]", "mono",
           "sinkage_pct_draught", "sinkage_pct_draught")
    figure("trim_s0.3.png", "Running trim, catamaran\nS/L = 0.3", "trim [deg, bow up]", "s0.3", "trim_deg", "trim_deg")

    for config in CONFIGS[1:]:
        def draw(ax, model, config=config):
            fns = [fn for fn in sorted(m_curves(model, "mono")) if fn >= 0.3]
            t = [(fn, tau(m_curves, model, config, "CWP_e3", fn)) for fn in fns]
            ax.plot(*zip(*[p for p in t if p[1] is not None]), **dots) if any(p[1] for p in t) else None
            for closure in closures:
                t = [(fn, tau(c_curves(closure), model, config, "CW_e3", fn)) for fn in fns]
                t = [p for p in t if p[1] is not None]
                if t:
                    ax.plot(*zip(*t), **line(closure))
            ax.axhline(1.0, color=MUTED, linewidth=0.8)
            ax.set_ylim(0, 3)
        if any((m, config, c) in computed for m in MODELS for c in closures):
            small_multiples(f"tau_{config}.png",
                            f"Wave interference, S/L = {config[1:]}:\ncatamaran / monohull\nwave coefficient",
                            "tau", draw,
                            [("measured C_WP ratio", dots)] + [(f"model, closure '{c}'", line(c)) for c in closures])

    ### Tables
    out = ["# Southampton catamaran series (Report 71): HydroSandbox against measurement", ""]
    done = sorted({(k[1], k[2]) for k in computed})
    out += [f"Curves run: {len(computed)} of {len(MODELS) * len(CONFIGS) * max(len(closures), 1)} "
            f"(configurations x closures present: {', '.join('/'.join(d) for d in done)}).", ""]

    def stats(rows, relative):
        if not rows:
            return "-"
        m, c = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
        if relative:
            return f"{np.median(c / m):.2f}"
        return f"{np.mean(c - m):+.2f} / {np.sqrt(np.mean((c - m) ** 2)):.2f}"

    def table(title, note, config, m_key, c_key, relative, closures_shown):
        lines = [f"## {title}", "", note, ""]
        head = "| model | " + " | ".join(f"{cl}, Fn {b[0]}" for cl in closures_shown for b in BANDS) + " |"
        lines += [head, "|" + "---|" * (1 + len(closures_shown) * len(BANDS))]
        for model in MODELS:
            cells = [stats(pairs(measured, computed, model, config, cl, m_key, c_key, b[1], b[2]), relative)
                     for cl in closures_shown for b in BANDS]
            if any(c != "-" for c in cells):
                lines.append(f"| {model} | " + " | ".join(cells) + " |")
        return lines + [""]

    for config in ("mono", "s0.3"):
        label = "monohull" if config == "mono" else f"catamaran S/L {config[1:]}"
        out += table(f"Wave resistance, {label}: model C_W / measured C_WP",
                     "Median ratio in each Froude band (1.00 is perfect).", config, "CWP_e3", "CW_e3", True, closures)
        out += table(f"Total resistance, {label}: model C_T / measured C_T",
                     "Median ratio in each Froude band.", config, "CT_e3", "CT_e3", True, closures)
        out += table(f"Trim, {label}: model - measured [deg]", "Mean error / rms error.",
                     config, "trim_deg", "trim_deg", False, closures)
        out += table(f"Sinkage, {label}: model - measured [% of draught]", "Mean error / rms error.",
                     config, "sinkage_pct_draught", "sinkage_pct_draught", False, closures)

    ### The non-wave remainder: measured C_T - C_WP against the model's friction + transom drag.
    out += ["## What is not wave pattern, monohull: (C_T - C_WP) / C_F(ITTC-57)", "",
            "Median over Fn 0.40-1.00. `measured` is the report's data; `Table III` is the form factor "
            "the report fits to it; `model` is (friction + transom drag) / C_F for each closure.", "",
            "| model | measured | Table III (1 + k) | " + " | ".join(f"model, {c}" for c in closures) + " |",
            "|" + "---|" * (3 + len(closures))]
    for model in MODELS:
        m = measured.get((model, "mono"), {})
        cells = []
        for closure in closures:
            c = computed.get((model, "mono", closure), {})
            v = [(c[fn]["CV_e3"] + c[fn]["Ctransom_e3"]) / (1e3 * ittc57(c[fn]["reynolds"]))
                 for fn in c if fn >= 0.4]
            cells.append(f"{np.median(v):.2f}" if v else "-")
        any_c = next((computed[k] for k in computed if k[0] == model), None)
        if not any_c:
            continue
        ratio = [(m[fn]["CT_e3"] - m[fn]["CWP_e3"]) / (1e3 * ittc57(any_c[fn]["reynolds"]))
                 for fn in m if fn >= 0.4 and fn in any_c]
        out.append(f"| {model} | {np.median(ratio):.2f} | {FORM_FACTORS[model][0]:.2f} | " + " | ".join(cells) + " |")
    out.append("")

    ### Superposition: interference factor, model against measured.
    out += ["## Wave interference factor tau = C_WP(cat) / C_WP(mono): model against measured", "",
            "Over Fn 0.40-1.00, all models with both curves: median measured tau, median model tau, and the rms "
            "of (model tau - measured tau). This tests the *addition* of two demihulls' waves separately from "
            "the size of one hull's.", "",
            "| S/L | closure | n | measured | model | rms difference |", "|---|---|---|---|---|---|"]
    for config in CONFIGS[1:]:
        for closure in closures:
            rows = []
            for model in MODELS:
                for fn in sorted(m_curves(model, config)):
                    if fn < 0.4:
                        continue
                    a = tau(m_curves, model, config, "CWP_e3", fn)
                    b = tau(c_curves(closure), model, config, "CW_e3", fn)
                    if a is not None and b is not None:
                        rows.append((a, b))
            if rows:
                a, b = np.array(rows).T
                out.append(f"| {config[1:]} | {closure} | {len(rows)} | {np.median(a):.2f} | {np.median(b):.2f} | "
                           f"{np.sqrt(np.mean((b - a) ** 2)):.2f} |")
    out.append("")
    failures = [(p.stem, r["Fn"]) for p in sorted((HERE / "results").glob("*.json"))
                for r in json.loads(p.read_text()) if "error" in r]
    out += [f"Solves that found no equilibrium: {len(failures)}"
            + (": " + ", ".join(f"{n} Fn {fn}" for n, fn in failures[:30]) if failures else "") + "."]
    (OUT / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
