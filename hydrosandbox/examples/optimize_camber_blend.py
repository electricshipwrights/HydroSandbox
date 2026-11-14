"""
Choose a hull from a camber family by optimization.

camber (github.com/electricshipwrights/camber) draws hulls whose authored numbers
blend: any convex combination of a family of designs is again a valid hull. Here
the blend weights are design variables of the same ``asb.Opti`` problem as the
rest of the craft (``optimize_foils_on_hull.optimize``): the hull's attitude and
loading, and optionally a main T-foil. The hull inside the solve is the family's
polynomial interpolant across exactly-swept blends (``HullFamily``); the solution
is then re-evaluated on the exact blend, and written out as a camber document.

    python hydrosandbox/examples/optimize_camber_blend.py a.json b.json \\
        --velocity 1.5 --cg-range=-0.3,0.3 --out best.json

Pass ``--foils`` to add a main T-foil (and ``--stabilizer`` a second one) to the
design; each starting layout is solved and the best kept.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydrosandbox as hsb  # noqa: E402
from hydrosandbox.hull.camber import HullFamily  # noqa: E402
from optimize_foils_on_hull import optimize, scalar, starting_layouts  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("documents", nargs="+", help="camber hull documents (2 or more)")
    parser.add_argument("--velocity", type=float, default=2.0, help="speed [m/s]")
    parser.add_argument(
        "--mass", type=float, default=None,
        help="all-up mass [kg] (default: what the family's centroid blend displaces)",
    )
    parser.add_argument(
        "--lcg", type=float, default=None,
        help="CG [m aft of the hull's bow] (default: the centroid blend's LCB)",
    )
    parser.add_argument("--vcg", type=float, default=0.1, help="CG height above the waterline [m]")
    parser.add_argument("--cg-range", type=str, default=None, help="allowed CG shift 'lo,hi' [m, aft]")
    parser.add_argument("--degree", type=int, default=None, help="interpolant degree in the weights")
    parser.add_argument("--foils", action="store_true", help="add a main T-foil to the design")
    parser.add_argument("--stabilizer", action="store_true", help="and a stabilizer T-foil")
    parser.add_argument("--out", type=str, default=None, help="write the chosen blend as a camber document")
    args = parser.parse_args()
    if len(args.documents) < 2:
        parser.error("a family needs at least two documents")

    t0 = time.time()
    family = HullFamily(args.documents, degree=args.degree)
    print(f"{family}: built in {time.time() - t0:.1f} s")
    water = hsb.Water.seawater()
    centroid = family.at([1.0 / family.n_hulls] * family.n_hulls)
    statics = centroid.hydrostatics()
    mass = args.mass or scalar(statics["volume"]) * water.density()
    x_bow = scalar(centroid.stations[0])
    x_cg = x_bow + args.lcg if args.lcg is not None else scalar(statics["x_cb"])
    cg_range = None if args.cg_range is None else tuple(float(v) for v in args.cg_range.split(","))

    starts = starting_layouts(centroid, args.stabilizer) if args.foils else [None]
    best = None
    for i, start in enumerate(starts):
        t0 = time.time()
        outcome = optimize(
            family, args.velocity, mass, [x_cg, 0.0, args.vcg], gml_floor=None, start=start,
            stabilizer=args.stabilizer, water=water, cg_range=cg_range,
        )
        label = f"start {i + 1}/{len(starts)}"
        if outcome is None:
            print(f"{label}: no solution ({time.time() - t0:.0f} s)")
            continue
        print(
            f"{label}: weights {[round(w, 4) for w in outcome['blend_weights']]}, "
            f"D = {scalar(outcome['results']['D']):.1f} N ({time.time() - t0:.0f} s)"
        )
        if best is None or outcome["results"]["D"] < best["results"]["D"]:
            best = outcome
    if best is None:
        sys.exit("No start converged.")

    r, load = best["results"], best["loading"]
    names = family.member_names
    print("\nChosen blend:")
    for name, w in zip(names, best["blend_weights"]):
        print(f"  {w:6.1%}  {name}")
    print(
        f"On the exact blend: D = {scalar(r['D']):.1f} N (hull friction "
        f"{scalar(r['D_hull_viscous']):.1f}, waves {scalar(r['D_wave']):.1f}"
        + (f", foils {scalar(r['D_foil']):.1f}" if "D_foil" in r else "")
        + f"), trim {r['trim']:+.2f} deg, sinkage {1000 * r['sinkage']:+.0f} mm, "
        f"F_z residual {scalar(r['F_z_net']):+.1f} N"
    )
    print(f"Loading: {load['mass']:.1f} kg, CG x = {load['x_cg']:.3f} m")
    if best["design"]:
        print("Design:", {k: round(v, 4) for k, v in best["design"].items()})
    if args.out:
        family.document(best["blend_weights"], name="optimized blend").to_json(args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
