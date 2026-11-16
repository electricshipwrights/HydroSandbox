"""
Feasibility test for a learned wave-spectrum correction.

    uv run python validation/southampton/spectrum_gain.py      (after spectra.py)

Hypothesis: one demihull's thin-ship wave spectrum, multiplied by a smooth
energy gain `G2(theta; Fn, hull form)` that knows nothing about the other hull,
gives the right wave-pattern resistance for the monohull *and*, through the
linear placement factor, for the catamaran at every spacing.

The measurements are Report 71's wave-pattern coefficients; the spectra are
thin ship at the measured sinkage and trim (`spectra.py`). Predictors compared,
all scored on `log(predicted / measured)` for Fn >= 0.3:

- `thin ship`: no correction.
- `mono-scaled`: each catamaran point scaled by its own monohull's measured /
  thin-ship ratio at that speed. The right total energy per hull, thin ship's
  distribution over theta: what a perfect drag-residual correction would give.
- `flat gain`: `G2` independent of theta, a smooth function of Fn and form.
- `theta gain`: `G2` piecewise linear in theta (nodes at 0, 30, 60, 90 deg).

The fitted gains are trained two ways: on monohull data only (then asked for
the catamarans), and on everything but one model (then asked for that model:
leave-one-model-out). Writes `report/spectrum_gain.md` and `.png`.

A pair of identical side-by-side hulls tests how the energy is distributed
over theta, not the phase of the amplitude: the phase cancels in
`|A|^2 * 4 cos^2(ky S / 2)`.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from npl import MODELS

FLOOR = 0.05  # [C x 1000] keeps the log error finite where there is almost no wave resistance
FN_MIN = 0.3
THETA = np.arange(90) + 0.5
RIDGE = 0.05


def hats(nodes):
    """Piecewise-linear basis over theta, (len(nodes), 90)."""
    nodes = np.asarray(nodes, dtype=float)
    if len(nodes) == 1:
        return np.ones((1, 90))
    return np.array([np.interp(THETA, nodes, np.eye(len(nodes))[j]) for j in range(len(nodes))])


def features(fn, model):
    x = (fn - 0.65) / 0.35
    s = np.array([(MODELS[m]["slenderness"] - 8.0) / 1.6 for m in model])
    b = np.array([(MODELS[m]["BT"] - 2.0) / 0.5 for m in model])
    return np.stack([np.ones_like(x), x, x**2, x**3, s, b, x * s, x * b], axis=1)


class Gain:
    def __init__(self, nodes):
        self.basis = hats(nodes)

    def log_gain(self, w, feats):
        return (feats @ w.reshape(feats.shape[1], len(self.basis))) @ self.basis  # (n, 90)

    def predict(self, w, d):
        return np.sum(d["energy"] * np.exp(self.log_gain(w, d["features"])), axis=1)

    def fit(self, d):
        n = d["features"].shape[1] * len(self.basis)
        target = np.log(d["CWP"] + FLOOR)
        residual = lambda w: np.concatenate([np.log(self.predict(w, d) + FLOOR) - target, RIDGE * w])
        return least_squares(residual, np.zeros(n)).x


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}


def score(pred, meas):
    e = np.log((pred + FLOOR) / (meas + FLOOR))
    return f"{np.exp(np.median(e)):.2f} / x{np.exp(np.sqrt(np.mean(e**2))):.2f}"


def main():
    raw = np.load(HERE / "spectra.npz")
    out = ["# Can a per-hull spectrum gain fix thin ship's wave-pattern resistance?", "",
           "Southampton series, Fn >= 0.3, thin ship at the measured attitude. Each cell: median predicted / "
           "measured, and the typical error factor (exp of the rms log error; x1.00 is perfect). "
           "`n` counts measured points.", ""]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True, facecolor="#fcfcfb")
    for closure in sorted(set(raw["closure"])):
        keep = (raw["closure"] == closure) & (raw["Fn"] >= FN_MIN - 1e-9)
        d = dict(model=raw["model"][keep], config=raw["config"][keep], Fn=raw["Fn"][keep],
                 CWP=raw["CWP_e3"][keep], energy=raw["energy"][keep])
        d["features"] = features(d["Fn"], d["model"])
        mono, cat = d["config"] == "mono", d["config"] != "mono"
        thin = d["energy"].sum(axis=1)

        # each catamaran point scaled by its monohull's measured / thin-ship ratio
        ratio = {(m, f): c / t for m, f, c, t in zip(d["model"][mono], d["Fn"][mono], d["CWP"][mono], thin[mono])}
        has_mono = np.array([(m, f) in ratio for m, f in zip(d["model"], d["Fn"])])
        scaled = np.array([thin[i] * ratio.get((d["model"][i], d["Fn"][i]), np.nan) for i in range(len(thin))])

        predictions = {}
        for name, nodes in (("flat gain", [0.0]), ("theta gain", [0.0, 30.0, 60.0, 90.0])):
            gain = Gain(nodes)
            predictions[name, "mono only"] = gain.predict(gain.fit(subset(d, mono)), d)
            held = np.zeros(len(thin))
            for model in MODELS:
                test = d["model"] == model
                held[test] = gain.predict(gain.fit(subset(d, ~test)), subset(d, test))
            predictions[name, "leave one model out"] = held
            if name == "theta gain":
                w_all = gain.fit(d)
                shown = gain

        out += [f"## Transom closure `{closure}`", "",
                f"| predictor | trained on | monohull (n={mono.sum()}) | catamaran (n={cat.sum()}) |", "|---|---|---|---|",
                f"| thin ship | - | {score(thin[mono], d['CWP'][mono])} | {score(thin[cat], d['CWP'][cat])} |",
                f"| mono-scaled | the same hull's monohull point | exact by construction | "
                f"{score(scaled[cat & has_mono], d['CWP'][cat & has_mono])} |"]
        for (name, training), pred in predictions.items():
            first = "(training data)" if training == "mono only" else score(pred[mono], d["CWP"][mono])
            out.append(f"| {name} | {training} | {first} | {score(pred[cat], d['CWP'][cat])} |")
        out += ["", "Leave-one-model-out theta gain, catamarans, by speed and by spacing:", "",
                "| | thin ship | mono-scaled | theta gain (held out) |", "|---|---|---|---|"]
        held = predictions["theta gain", "leave one model out"]
        groups = [(f"Fn {lo:.2f}-{hi:.2f}", cat & (d["Fn"] >= lo - 1e-9) & (d["Fn"] <= hi + 1e-9))
                  for lo, hi in ((0.3, 0.45), (0.5, 0.7), (0.75, 1.0))]
        groups += [(f"S/L {c[1:]}", d["config"] == c) for c in ("s0.2", "s0.3", "s0.4", "s0.5")]
        for label, g in groups:
            out.append(f"| {label} | {score(thin[g], d['CWP'][g])} | {score(scaled[g & has_mono], d['CWP'][g & has_mono])} "
                       f"| {score(held[g], d['CWP'][g])} |")
        out.append("")
        if closure == "none":
            out += ["Fitted energy gain G2 (all data), model 5b:", "", "| Fn | theta 0 | 30 | 60 | 90 deg |", "|---|---|---|---|---|"]
            for fn in (0.3, 0.4, 0.5, 0.7, 1.0):
                g = np.exp(features(np.array([fn]), ["5b"]) @ w_all.reshape(-1, len(shown.basis)))[0]
                out.append(f"| {fn} | " + " | ".join(f"{v:.2f}" for v in g) + " |")
            out.append("")
            for i, model in enumerate(("3b", "5b", "6b")):
                for j, config in enumerate(("mono", "s0.2", "s0.4")):
                    ax, m = axes[i, j], (d["model"] == model) & (d["config"] == config)
                    o = np.argsort(d["Fn"][m])
                    ax.plot(d["Fn"][m][o], d["CWP"][m][o], color="#0b0b0b", marker="o", markersize=4, linestyle="none",
                            label="measured wave pattern (Report 71)")
                    ax.plot(d["Fn"][m][o], thin[m][o], color="#2a78d6", linewidth=2, label="thin ship, no transom closure")
                    ax.plot(d["Fn"][m][o], held[m][o], color="#eb6834", linewidth=2,
                            label="thin ship x learned gain (this hull held out)")
                    ax.set_title(f"Model {model}, {'monohull' if config == 'mono' else 'catamaran S/L ' + config[1:]}",
                                 loc="left", fontsize=10)
                    ax.set_facecolor("#fcfcfb"); ax.grid(True, color="#e1e0d9", linewidth=0.6)
                    ax.tick_params(colors="#898781", labelsize=8); ax.set_ylim(0, None)
                    for side in ("top", "right"):
                        ax.spines[side].set_visible(False)
                    if j == 0:
                        ax.set_ylabel("C_WP x 1000", fontsize=9)
                    if i == 2:
                        ax.set_xlabel("Froude number", fontsize=9)
            axes[0, 0].legend(frameon=False, fontsize=8)
    (HERE / "report").mkdir(exist_ok=True)
    fig.suptitle("Wave-pattern resistance: thin ship, and thin ship with a per-hull spectrum gain", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(); fig.savefig(HERE / "report" / "spectrum_gain.png", dpi=110)
    (HERE / "report" / "spectrum_gain.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
