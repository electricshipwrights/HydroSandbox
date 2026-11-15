"""
A catamaran demihull designed from scratch as a camber document: no drawn hull,
just the generator's numbers (``hydrosandbox.hull.camber.generate``), searched
for the least drag at a cruise speed under limits on pitch stiffness, draft,
freeboard and reserve buoyancy.

Each candidate is an exact evaluation: generate the document, set its waterline
for the displacement, build the hull, find the cat's equilibrium at speed and its
pitch stiffness there and near rest. The search runs in parallel on a
penalized objective: differential evolution, then a pattern search to polish
(derivative-free: the hull table is rebuilt per candidate).

    python -m hydrosandbox.examples.design_camber_demihull --jobs 8

Defaults are a NACRA-sized beach cat: 5.5 m, 500 kg, 2.2 m spacing, 6 m/s.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict

import numpy as onp

import hydrosandbox as hsb
from hydrosandbox.hull import CraftBuildup, Hull
from hydrosandbox.hull.camber.generate import DemihullParameters, demihull, with_displacement

GRID = dict(n_stations=61, n_levels=41)
WAVE = (3.0, 8.0, 0.01)

#: The searched numbers and their ranges; per-station ones are ``name.i``.
PLAN_BOUNDS = {
    "transom_half_beam": (100.0, 300.0),
    "mid_half_beam": (120.0, 300.0),
    "x_mid": (0.25, 0.70),
    "entry_angle": (3.0, 20.0),
    "flare": (0.0, 60.0),
    "keel_reach": (150.0, 450.0),
}
STATION_BOUNDS = {"side": (150.0, 600.0), "bottom": (40.0, 400.0), "fullness": (0.0, 1.0)}
#: Interior stations' positions (plan parameter), by station count.
U_BOUNDS = {3: [(0.15, 0.85)], 4: [(0.10, 0.50), (0.50, 0.90)]}


def bounds(n_stations: int = 3) -> dict:
    """Every searched number and its range, for hulls of ``n_stations`` stations."""
    out = dict(PLAN_BOUNDS)
    out.update({f"station_u.{i + 1}": r for i, r in enumerate(U_BOUNDS[n_stations])})
    for name, r in STATION_BOUNDS.items():
        out.update({f"{name}.{i}": r for i in range(n_stations)})
    return out


def nest(flat: dict) -> dict:
    """Flat ``name.i`` numbers to ``DemihullParameters`` fields (station_u gets its ends)."""
    out, per = {}, {}
    for k, v in flat.items():
        if "." in k:
            name, i = k.split(".")
            per.setdefault(name, {})[int(i)] = float(v)
        else:
            out[k] = float(v)
    for name, d in per.items():
        if name == "station_u":
            out[name] = (0.0, *[d[i] for i in sorted(d)], 1.0)
        else:
            out[name] = tuple(d[i] for i in sorted(d))
    return out


def flatten(params: dict, keys) -> dict:
    """``DemihullParameters`` fields (as from ``asdict``) to the flat numbers ``keys``."""
    out = {}
    for k in keys:
        if "." in k:
            name, i = k.split(".")
            out[k] = float(params[name][int(i)])
        else:
            out[k] = float(params[k])
    return out


def _f(v) -> float:
    return float(onp.ravel(v)[0])


def evaluate_hull(hull: Hull, mass: float, spacing: float, speed: float, rest_speed: float = 1.0) -> dict:
    """Drag, attitude and pitch stiffness of a cat of two ``hull``s, CG over the LCB at the waterline."""
    water = hsb.Water.seawater()
    x_cb = _f(hull.hydrostatics()["x_cb"])
    common = dict(
        hull=hull, mass=mass, xyz_cg=[x_cb, 0.0, 0.0], water=water, demihull_spacing=spacing,
        wave_length_scales=WAVE, include_interference_breakdown=False, include_transom_drag=True,
    )
    out = {}
    for tag, v in (("cruise", speed), ("rest", rest_speed)):
        eq = CraftBuildup.solve_equilibrium(velocity=v, max_iterations=60, tolerance=2e-4, **common)
        r = CraftBuildup(velocity=v, sinkage=eq["sinkage"], trim=eq["trim"], **common).run_with_stability(
            include_pitch_damping=False
        )
        out[tag] = {
            "D": _f(r["D"]), "D_wave": _f(r["D_wave"]), "D_viscous": _f(r["D_hull_viscous"]),
            "D_transom": _f(r.get("D_transom", 0.0)), "trim": _f(r["trim"]), "sinkage": _f(r["sinkage"]),
            "GM_L": _f(r["GM_L"]), "k_heave": _f(r["k_heave"]),
        }
    return out


def geometry(hull: Hull) -> dict:
    """Draft, freeboard, waterline beam, wetted area, transom immersion and reserve buoyancy."""
    hb = onp.asarray(hull.half_breadths, dtype=float)
    st = onp.asarray(hull.stations, dtype=float)
    lv = onp.asarray(hull.levels, dtype=float)
    hs = hull.hydrostatics()
    volume = _f(hs["volume"])
    # Volume below the deck: the hull sunk to its top level.
    full = _f(hull.hydrostatics(float(lv[-1]) - 1e-6, 0.0, (0.0, 0.0))["volume"])
    # Above-water volume forward of amidships (resists burying the bows).
    above = lv >= 0.0
    fwd = st <= st[0] + 0.5 * (st[-1] - st[0])
    area = 2.0 * onp.trapezoid(hb[:, above], lv[above], axis=1)
    reserve_fwd = float(onp.trapezoid(area[fwd], st[fwd]))
    wl = onp.array([onp.interp(0.0, lv, row) for row in hb])
    keel_aft = float(hull.keel_height(st[-1]))
    return {
        "length": float(hull.length), "volume": volume, "draft": -float(lv[0]), "freeboard": float(lv[-1]),
        "beam_wl": 2.0 * float(wl.max()), "slenderness": float(hull.length) / (2.0 * float(wl.max())),
        "wetted_surface": _f(hs["wetted_surface"]), "x_cb": _f(hs["x_cb"]) - float(st[0]),
        "transom_immersion": max(0.0, -keel_aft), "reserve_ratio": full / volume, "reserve_fwd": reserve_fwd,
    }


def evaluate(params: dict, mass: float = 500.0, spacing: float = 2.2, speed: float = 6.0) -> dict:
    """One candidate: its document, geometry and performance (or the error that stopped it)."""
    t0 = time.time()
    p = DemihullParameters().replace(**nest(params))
    res = {"params": asdict(p)}
    try:
        doc = with_displacement(demihull(p), 0.5 * mass / hsb.Water.seawater().density(), **GRID)
        hull = Hull.from_camber(doc, **GRID)
        res["geometry"] = geometry(hull)
        res.update(evaluate_hull(hull, mass, spacing, speed))
        res["document"] = doc.to_dict()
    except Exception as e:  # an invalid shape or no equilibrium: infeasible
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    res["seconds"] = time.time() - t0
    return res


# -- the search --------------------------------------------------------------------
def violations(res: dict, limits: dict) -> dict:
    """Each limit's relative shortfall (0 when met)."""
    if "error" in res:
        return {"error": 1.0}
    g, c, r = res["geometry"], res["cruise"], res["rest"]
    v = {
        "power": c["D"] * limits["speed"] / limits["efficiency"] / limits["power"] - 1.0,
        "GM_L_cruise": 1.0 - c["GM_L"] / limits["GM_L_cruise"],
        "GM_L_rest": 1.0 - r["GM_L"] / limits["GM_L_rest"],
        "draft": g["draft"] / limits["draft"] - 1.0,
        "freeboard": 1.0 - g["freeboard"] / limits["freeboard"],
        "reserve_ratio": 1.0 - g["reserve_ratio"] / limits["reserve_ratio"],
        "reserve_fwd": 1.0 - g["reserve_fwd"] / limits["reserve_fwd"],
        "transom_immersion": (g["transom_immersion"] - limits["transom_immersion"]) / g["draft"],
    }
    return {k: max(0.0, x) for k, x in v.items()}


def merit(res: dict, limits: dict, weight: float = 20.0) -> float:
    """Drag [N] plus a penalty of ``weight`` x drag-scale per unit of relative violation."""
    v = violations(res, limits)
    if "error" in v:
        return 1e6
    return res["cruise"]["D"] * (1.0 + weight * sum(v.values()))


#: The search's ranges (``bounds(n)``; set by ``main``).
BOUNDS = bounds(3)


def to_unit(params: dict) -> onp.ndarray:
    return onp.array([(params[k] - lo) / (hi - lo) for k, (lo, hi) in BOUNDS.items()])


def from_unit(x) -> dict:
    return {k: float(lo + (hi - lo) * onp.clip(xi, 0.0, 1.0)) for xi, (k, (lo, hi)) in zip(x, BOUNDS.items())}


def pattern_search(start: dict, limits: dict, jobs: int, step: float = 0.15, min_step: float = 0.01,
                   max_rounds: int = 40, log=print, history: list | None = None, case: dict | None = None,
                   best_path: str | None = None):
    """
    Parallel compass search in the unit box: try +-step along every parameter at
    once, move to the best improvement (then try the same move again, doubled),
    halve the step when nothing improves.
    """
    case = case or {}
    history = history if history is not None else []
    run = lambda xs: hsb.parallel_map(evaluate, [(from_unit(x), *case.values()) for x in xs], jobs=jobs)
    x = to_unit(start)
    best = run([x])[0]
    history.append(best)
    f_best = merit(best, limits)
    log(f"start: merit {f_best:.1f}  {summary(best)}")
    for rnd in range(max_rounds):
        if step < min_step:
            break
        trials = []
        for i in range(len(x)):
            for s in (+step, -step):
                y = x.copy()
                y[i] = onp.clip(y[i] + s, 0.0, 1.0)
                if not onp.allclose(y, x):
                    trials.append(y)
        results = run(trials)
        history.extend(results)
        scores = [merit(r, limits) for r in results]
        j = int(onp.argmin(scores))
        if scores[j] < f_best - 1e-3:
            move = trials[j] - x
            x, best, f_best = trials[j], results[j], scores[j]
            # Try going further the same way.
            ahead = onp.clip(x + move, 0.0, 1.0)
            r2 = run([ahead])[0]
            history.append(r2)
            if merit(r2, limits) < f_best:
                x, best, f_best = ahead, r2, merit(r2, limits)
            save_best(best, best_path)
            log(f"round {rnd}: step {step:.3f}, merit {f_best:.1f}  {summary(best)}")
        else:
            step *= 0.5
            log(f"round {rnd}: no improvement, step -> {step:.3f}")
    return best, history


def save_best(res: dict, path: str | None) -> None:
    """Write a candidate's camber document (and its numbers) to ``path``, if given."""
    if path and "document" in res:
        json.dump(res["document"], open(path, "w"), indent=1)
        json.dump({k: res[k] for k in res if k != "document"}, open(path.replace(".camber.json", ".best.json"), "w"), indent=1)


def evolve(start: dict, limits: dict, jobs: int, generations: int = 40, popsize: int = 4, seed: int = 0,
           log=print, history: list | None = None, case: dict | None = None, best_path: str | None = None):
    """Differential evolution in the unit box (population ``popsize`` x parameters), in parallel."""
    from scipy.optimize import differential_evolution

    case = case or {}
    history = history if history is not None else []
    state = {"gen": 0, "best": None, "f": math.inf}

    def mapper(_func, xs):
        results = hsb.parallel_map(evaluate, [(from_unit(x), *case.values()) for x in xs], jobs=jobs)
        history.extend(results)
        scores = [merit(r, limits) for r in results]
        j = int(onp.argmin(scores))
        if scores[j] < state["f"]:
            state["best"], state["f"] = results[j], scores[j]
            save_best(results[j], best_path)
        log(f"generation {state['gen']}: {len(xs)} hulls, best merit {state['f']:.1f}  {summary(state['best'])}")
        state["gen"] += 1
        return scores

    differential_evolution(
        lambda x: 0.0, [(0.0, 1.0)] * len(BOUNDS), workers=mapper, updating="deferred", popsize=popsize,
        maxiter=generations, polish=False, init="latinhypercube", x0=to_unit(start), seed=seed, tol=1e-3,
        mutation=(0.5, 1.0), recombination=0.7,
    )
    return state["best"], history


def summary(res: dict) -> str:
    if "error" in res:
        return res["error"]
    g, c, r = res["geometry"], res["cruise"], res["rest"]
    return (
        f"D {c['D']:.1f} N (wave {c['D_wave']:.1f}, visc {c['D_viscous']:.1f}, transom {c['D_transom']:.1f}), "
        f"trim {c['trim']:+.2f}, GM_L {c['GM_L']:.1f}/{r['GM_L']:.1f} m, draft {g['draft']:.3f}, "
        f"fb {g['freeboard']:.3f}, B_wl {g['beam_wl']:.3f} (L/B {g['slenderness']:.1f}), S {g['wetted_surface']:.2f}, "
        f"transom {g['transom_immersion']:.3f}, reserve {g['reserve_ratio']:.2f}/{g['reserve_fwd']:.3f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass", type=float, default=500.0)
    ap.add_argument("--spacing", type=float, default=2.2)
    ap.add_argument("--speed", type=float, default=6.0)
    ap.add_argument("--power", type=float, default=4000.0, help="battery power at speed [W]")
    ap.add_argument("--efficiency", type=float, default=0.6, help="battery-to-thrust efficiency")
    ap.add_argument("--gml-cruise", type=float, required=True, help="min pitch GM_L at speed [m]")
    ap.add_argument("--gml-rest", type=float, required=True, help="min pitch GM_L near rest [m]")
    ap.add_argument("--draft", type=float, default=0.30)
    ap.add_argument("--freeboard", type=float, default=0.35)
    ap.add_argument("--reserve-ratio", type=float, required=True, help="min volume to deck / displacement")
    ap.add_argument("--reserve-fwd", type=float, required=True, help="min above-water volume, fwd half [m^3]")
    ap.add_argument("--transom-immersion", type=float, default=0.08, help="max at rest [m]")
    ap.add_argument("--stations", type=int, default=3, choices=(3, 4))
    ap.add_argument("--start", help="a .best.json from an earlier search to start from")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=40, help="pattern-search rounds (after evolution)")
    ap.add_argument("--generations", type=int, default=40, help="differential-evolution generations (0: none)")
    ap.add_argument("--popsize", type=int, default=3, help="population per searched number")
    ap.add_argument("--out", default="design_camber_demihull.json")
    a = ap.parse_args()
    limits = dict(
        speed=a.speed, power=a.power, efficiency=a.efficiency, GM_L_cruise=a.gml_cruise, GM_L_rest=a.gml_rest,
        draft=a.draft, freeboard=a.freeboard, reserve_ratio=a.reserve_ratio, reserve_fwd=a.reserve_fwd,
        transom_immersion=a.transom_immersion,
    )
    case = {"mass": a.mass, "spacing": a.spacing, "speed": a.speed}
    history: list = []
    best_path = a.out.replace(".json", ".camber.json")  # kept current as the search improves
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    global BOUNDS
    BOUNDS = bounds(a.stations)
    p0 = DemihullParameters()
    if a.stations == 4:
        p0 = p0.replace(station_u=(0.0, 0.3, 0.7, 1.0), side=(330.0, 400.0, 450.0, 480.0),
                        bottom=(180.0, 230.0, 230.0, 220.0), fullness=(0.6, 0.5, 0.45, 0.4))
    start = flatten(asdict(p0), BOUNDS)
    if a.start:
        start = flatten(json.load(open(a.start))["params"], BOUNDS)
    if a.generations:
        best, history = evolve(start, limits, a.jobs, a.generations, popsize=a.popsize, log=log, history=history, case=case,
                               best_path=best_path)
        start = flatten(best["params"], BOUNDS)
    best, history = pattern_search(start, limits, a.jobs, step=0.05 if a.generations else 0.15,
                                   max_rounds=a.rounds, log=log, history=history, case=case, best_path=best_path)
    log(f"best: {summary(best)}")
    log(f"violations: { {k: round(v, 4) for k, v in violations(best, limits).items() if v} or 'none'}")
    json.dump({"limits": limits, "case": case, "best": best, "history": history}, open(a.out, "w"), indent=1)
    save_best(best, best_path)


if __name__ == "__main__":
    main()
