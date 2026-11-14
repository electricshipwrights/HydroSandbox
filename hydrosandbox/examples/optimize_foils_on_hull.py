"""
Hull-in-the-loop design of lift-assist hydrofoils: minimum drag at speed, subject to
a floor on static pitch stability.

A hull (thin-ship theory, free to sink and trim) carries a main T-foil and,
optionally, a stabilizer T-foil. Everything is one ``asb.Opti`` problem:

- design variables: each foil's fore-aft position, depth, span, chords and incidence;
  and, within given ranges, the craft's loading: the fore-aft position of the centre
  of gravity, the all-up mass, and the height of the thrust line;
- state variables: the craft's sinkage and trim, held in vertical-force and
  pitching-moment balance by two equality constraints;
- objective: total drag (hull friction + foils + the wave resistance of the hull
  and foils together, interference included);
- stability: the heave-free pitch stiffness, as an effective longitudinal
  metacentric height ``GM_L = k_pitch / (m g)``, must stay above a floor.

Sweeping the floor traces the drag-versus-stability trade-off. The problem is not
convex (wave interference oscillates with foil position), so each floor is solved
from several starting layouts and the best feasible result kept. Those solves are
independent, and run on all cores at once (``--jobs``).

    python hydrosandbox/examples/optimize_foils_on_hull.py --velocity 4.5 \\
        --gml-fractions 0.6,0.8,1.0 --stabilizer --cg-range=-0.5,0.5 \\
        --thrust-height-range=-0.4,0.2

(Ranges that start with a minus sign need the ``--option=lo,hi`` form.)

Each case is reported against two baselines: the bare hull as loaded, and the bare
hull with the same loading freedoms optimized (so that what the foils buy is not
confused with what moving the centre of gravity buys on its own).

By default the hull is a slender Wigley demihull. ``--hull-record FILE`` reads the
first ``hull`` record of a boatmath JSON-lines stream instead
(``boatmath hull my.igs > hull.jsonl``), with ``--mass`` and ``--lcg`` (metres aft of
the bow) describing its load.
"""

import argparse
import json
import time
import numpy as onp
import aerosandbox as asb
import aerosandbox.numpy as np
import hydrosandbox as hsb
from hydrosandbox.hull import CraftBuildup, Hull


def scalar(value) -> float:
    return float(onp.ravel(value)[0])


def build_assembly(design: dict, stabilizer: bool) -> hsb.HydrofoilAssembly:
    """A main T-foil and, optionally, a stabilizer T-foil, from design variables."""
    main_airfoil = asb.Airfoil("naca2412")
    strut_airfoil = asb.Airfoil("naca0012")

    def t_foil(name, x, depth, half_span, root_chord, tip_chord, incidence, airfoil, role):
        foil = hsb.Foil(
            name=name,
            symmetric=True,
            role=role,
            xsecs=[
                asb.WingXSec(
                    xyz_le=[x, 0.0, -depth], chord=root_chord, twist=incidence, airfoil=airfoil
                ),
                asb.WingXSec(
                    xyz_le=[x + 0.25 * (root_chord - tip_chord), half_span, -depth],
                    chord=tip_chord,
                    twist=incidence,
                    airfoil=airfoil,
                ),
            ],
        )
        strut = hsb.Strut.from_dimensions(
            name=f"{name} strut",
            root_xyz=[x + 0.1 * root_chord, 0.0, -depth],
            length=depth,
            root_chord=0.8 * root_chord,
            tip_chord=0.8 * root_chord,
            airfoil=strut_airfoil,
        )
        return foil, strut

    main, main_strut = t_foil(
        "Main foil",
        design["x_main"],
        design["depth_main"],
        design["half_span_main"],
        design["root_chord_main"],
        design["taper_main"] * design["root_chord_main"],
        design["incidence_main"],
        main_airfoil,
        "main",
    )
    foils, struts = [main], [main_strut]
    if stabilizer:
        stab, stab_strut = t_foil(
            "Stabilizer",
            design["x_stab"],
            design["depth_stab"],
            design["half_span_stab"],
            design["chord_stab"],
            design["chord_stab"],
            design["incidence_stab"],
            asb.Airfoil("naca0012"),
            "stabilizer",
        )
        foils.append(stab)
        struts.append(stab_strut)
    return hsb.HydrofoilAssembly(name="Lift-assist foils", foils=foils, struts=struts)


def design_bounds(hull: Hull, stabilizer: bool) -> dict[str, tuple[float, float]]:
    x_bow, x_stern = float(hull.stations[0]), float(hull.stations[-1])
    length = x_stern - x_bow
    bounds = {
        "x_main": (x_bow + 0.10 * length, x_bow + 0.95 * length),
        "depth_main": (0.20, 0.50),
        "half_span_main": (0.15, 0.45),
        "root_chord_main": (0.04, 0.16),
        "taper_main": (0.40, 1.00),
        "incidence_main": (-3.0, 6.0),
    }
    if stabilizer:
        bounds.update(
            {
                "x_stab": (x_bow + 0.30 * length, x_bow + 1.00 * length),
                "depth_stab": (0.20, 0.50),
                "half_span_stab": (0.08, 0.30),
                "chord_stab": (0.03, 0.10),
                "incidence_stab": (-4.0, 4.0),
            }
        )
    return bounds


def starting_layouts(hull: Hull, stabilizer: bool) -> list[dict[str, float]]:
    x_bow = float(hull.stations[0])
    length = hull.length
    layouts = []
    for main_fraction, stab_fraction in ((0.30, 0.90), (0.50, 0.92), (0.72, 0.97)):
        layout = {
            "x_main": x_bow + main_fraction * length,
            "depth_main": 0.35,
            "half_span_main": 0.30,
            "root_chord_main": 0.09,
            "taper_main": 0.6,
            "incidence_main": 1.5,
        }
        if stabilizer:
            layout.update(
                {
                    "x_stab": x_bow + stab_fraction * length,
                    "depth_stab": 0.35,
                    "half_span_stab": 0.15,
                    "chord_stab": 0.05,
                    "incidence_stab": 0.0,
                }
            )
        layouts.append(layout)
    return layouts


def optimize(
    hull: Hull,
    velocity: float,
    mass: float,
    xyz_cg: list[float],
    gml_floor: float | None,
    start: dict[str, float] | None,
    stabilizer: bool,
    water: hsb.Water,
    cg_range: tuple[float, float] | None = None,
    mass_range: tuple[float, float] | None = None,
    thrust_height_range: tuple[float, float] | None = None,
    trim_bounds: tuple[float, float] = (-1.0, 3.0),
    max_iterations: int = 120,
    craft_options: dict | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict | None:
    """
    One optimization from one starting layout (``start=None``: the bare hull, with
    only its loading free). Returns None if it does not converge.

    ``cg_range`` is the allowed fore-aft shift of the centre of gravity [m, aft
    positive], ``mass_range`` the allowed all-up mass [kg], and
    ``thrust_height_range`` the allowed height of the thrust line at the stern
    [m above the design waterline]. Each is fixed if None (thrust then acts through
    the centre of gravity). ``craft_options`` are further ``CraftBuildup`` arguments
    (e.g. ``demihull_spacing`` for a catamaran); ``bounds`` overrides entries of
    ``design_bounds``.
    """
    with_foils = start is not None
    x_stern = float(hull.stations[-1])
    thrust_start = (
        None
        if thrust_height_range is None
        else [x_stern, 0.0, float(onp.clip(xyz_cg[2], *thrust_height_range))]
    )
    common = dict(
        hull=hull,
        velocity=velocity,
        water=water,
        wave_length_scales=(1.0, 1.5 * hull.length, 0.01),
        include_interference_breakdown=False,
    )
    common.update(craft_options or {})
    try:
        at_start = CraftBuildup.solve_equilibrium(
            assembly=build_assembly(start, stabilizer) if with_foils else None,
            mass=mass,
            xyz_cg=xyz_cg,
            xyz_thrust=thrust_start,
            **common,
        )
    except RuntimeError:
        return None

    opti = asb.Opti()
    design = {}
    if with_foils:
        design = {
            name: opti.variable(init_guess=start[name], lower_bound=lo, upper_bound=hi)
            for name, (lo, hi) in {**design_bounds(hull, stabilizer), **(bounds or {})}.items()
        }
    loading = {"x_cg": xyz_cg[0], "mass": mass, "thrust_height": None}
    if cg_range is not None:
        loading["x_cg"] = opti.variable(
            init_guess=xyz_cg[0],
            lower_bound=xyz_cg[0] + cg_range[0],
            upper_bound=xyz_cg[0] + cg_range[1],
        )
    if mass_range is not None:
        loading["mass"] = opti.variable(
            init_guess=mass, lower_bound=mass_range[0], upper_bound=mass_range[1]
        )
    if thrust_height_range is not None:
        loading["thrust_height"] = opti.variable(
            init_guess=thrust_start[2],
            lower_bound=thrust_height_range[0],
            upper_bound=thrust_height_range[1],
        )

    def craft_inputs(values: dict) -> dict:
        return dict(
            mass=values["mass"],
            xyz_cg=[values["x_cg"], 0.0, xyz_cg[2]],
            xyz_thrust=(
                None
                if values["thrust_height"] is None
                else [x_stern, 0.0, values["thrust_height"]]
            ),
        )
    sinkage = opti.variable(init_guess=at_start["sinkage"], lower_bound=-0.4, upper_bound=0.2)
    trim = opti.variable(
        init_guess=float(onp.clip(at_start["trim"], *trim_bounds)),
        lower_bound=trim_bounds[0],
        upper_bound=trim_bounds[1],
    )

    assembly = build_assembly(design, stabilizer) if with_foils else None
    craft = CraftBuildup(
        assembly=assembly, sinkage=sinkage, trim=trim, **craft_inputs(loading), **common
    )
    res = craft.run_with_stability() if gml_floor is not None else craft.run()
    weight = mass * water.gravity
    q = 0.5 * water.density() * velocity**2

    constraints = [
        res["F_z_net"] / weight == 0,
        res["M_y_net"] / (weight * hull.length) == 0,
    ]
    if with_foils:
        foil = res["foil"]
        constraints += [
            foil["cavitation_margin"] >= 0.3,
            foil["ventilation_margin"] >= 0.1,
            # Main foil: aspect ratio within a sensible structural limit.
            2.0 * design["half_span_main"]
            <= 10.0 * 0.5 * (1.0 + design["taper_main"]) * design["root_chord_main"],
        ]
        # Foil loading within stall limits.
        for wing, component in zip(assembly.wings, foil["wing_hydro_components"]):
            if not getattr(wing, "is_strut", False):
                lift_coefficient = component.L / (q * wing.area())
                constraints += [lift_coefficient <= 0.7, lift_coefficient >= -0.5]
        if stabilizer:
            constraints.append(design["x_stab"] >= design["x_main"] + 0.5)
    if gml_floor is not None:
        constraints.append(res["GM_L"] / gml_floor >= 1.0)

    opti.subject_to(constraints)
    opti.minimize(res["D"] / (0.1 * weight))
    try:
        sol = opti.solve(
            verbose=False,
            options={
                "ipopt.hessian_approximation": "limited-memory",
                "ipopt.max_iter": max_iterations,
            },
        )
    except RuntimeError:
        return None

    def value(v):
        return v if v is None or isinstance(v, (int, float)) else float(sol(v))

    solved = {name: float(sol(var)) for name, var in design.items()}
    solved_loading = {name: value(v) for name, v in loading.items()}
    # Re-evaluate numerically at the solution, with the interference breakdown.
    final = CraftBuildup(
        assembly=build_assembly(solved, stabilizer) if with_foils else None,
        sinkage=float(sol(sinkage)),
        trim=float(sol(trim)),
        **craft_inputs(solved_loading),
        **dict(common, include_interference_breakdown=with_foils),
    ).run_with_stability()
    # Keep the returned results plain (numbers only), so they can cross processes.
    final = {
        key: (scalar(value) if key not in ("sinkage", "trim") else float(value))
        for key, value in final.items()
        if key not in ("foil", "transom", "xyz_cg") and value is not None
    }
    return {"design": solved, "loading": solved_loading, "results": final}


def optimize_task(task: tuple, freedoms: dict) -> dict | None:
    """``optimize`` with its loading freedoms as one argument, for ``parallel_map``."""
    return optimize(*task, **freedoms)


def describe(label: str, outcome: dict, hull: Hull) -> str:
    d, r, load = outcome["design"], outcome["results"], outcome["loading"]
    weight = load["mass"] * 9.80665
    x_bow = float(hull.stations[0])
    thrust = (
        "through the CG"
        if load["thrust_height"] is None
        else f"{load['thrust_height']:+.2f} m from the waterline"
    )
    line = (
        f"{label}: D = {scalar(r['D']):6.1f} N, GM_L = {scalar(r['GM_L']):5.1f} m, "
        f"trim {r['trim']:+.2f} deg, sinkage {1000 * r['sinkage']:+.0f} mm\n"
        f"      loading: mass {load['mass']:.1f} kg, CG {load['x_cg'] - x_bow:.2f} m aft of "
        f"the bow, thrust line {thrust}\n"
        f"      drag: hull friction {scalar(r['D_hull_viscous']):.1f} + foils "
        f"{scalar(r['D_foil']):.1f} + waves {scalar(r['D_wave']):.1f} N"
    )
    if d:
        line += (
            f" (hull-foil interference {scalar(r['D_wave_interference']):+.1f}); "
            f"foils carry {scalar(r['L_foil']) / weight:.0%} of the weight\n"
            f"      main: {d['x_main'] - x_bow:.2f} m aft of the bow, depth "
            f"{d['depth_main']:.2f} m, span {2 * d['half_span_main']:.2f} m, chord "
            f"{d['root_chord_main']:.3f}->{d['taper_main'] * d['root_chord_main']:.3f} m, "
            f"incidence {d['incidence_main']:+.2f} deg"
        )
    if "x_stab" in d:
        line += (
            f"\n      stabilizer: {d['x_stab'] - x_bow:.2f} m aft of the bow, depth "
            f"{d['depth_stab']:.2f} m, span {2 * d['half_span_stab']:.2f} m, chord "
            f"{d['chord_stab']:.3f} m, incidence {d['incidence_stab']:+.2f} deg"
        )
    return line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--velocity", type=float, default=4.5, help="speed [m/s]")
    parser.add_argument("--hull-record", type=str, default=None, help="boatmath hull JSONL")
    parser.add_argument("--mass", type=float, default=None, help="all-up mass [kg]")
    parser.add_argument("--lcg", type=float, default=None, help="CG, metres aft of the bow")
    parser.add_argument("--vcg", type=float, default=0.0, help="CG height above the waterline [m]")
    parser.add_argument(
        "--gml-fractions",
        type=str,
        default="0.6,0.8,1.0",
        help="pitch-stability floors, as fractions of the bare hull's GM_L at speed",
    )
    parser.add_argument("--stabilizer", action="store_true", help="add a stabilizer T-foil")
    parser.add_argument(
        "--cg-range", type=str, default=None,
        help="allowed fore-aft CG shift 'lo,hi' [m, aft positive], e.g. --cg-range=-0.5,0.5",
    )
    parser.add_argument(
        "--mass-range", type=str, default=None, help="allowed all-up mass 'lo,hi' [kg]"
    )
    parser.add_argument(
        "--thrust-height-range", type=str, default=None,
        help="allowed thrust-line height at the stern 'lo,hi' [m above the waterline]",
    )
    parser.add_argument("--starts", type=int, default=3, help="starting layouts per floor (1-3)")
    parser.add_argument(
        "--jobs", type=int, default=None, help="worker processes (default: all cores)"
    )
    args = parser.parse_args()

    water = hsb.Water.seawater()
    if args.hull_record:
        with open(args.hull_record) as f:
            record = next(r for r in map(json.loads, f) if r.get("type") == "hull")
        hull = Hull.from_boatmath_geometry(
            record["geometry"], n_stations=61, n_levels=41, name=record.get("name") or "hull"
        )
    else:
        hull = Hull.wigley(5.0, 0.42, 0.12, freeboard=0.35, n_stations=41, n_levels=25)
    statics = hull.hydrostatics()
    mass = args.mass or scalar(statics["volume"]) * water.density()
    x_cg = (
        float(hull.stations[0]) + args.lcg if args.lcg is not None else scalar(statics["x_cb"])
    )
    xyz_cg = [x_cg, 0.0, args.vcg]
    weight = mass * water.gravity

    common = dict(
        hull=hull, velocity=args.velocity, mass=mass, xyz_cg=xyz_cg, water=water,
        wave_length_scales=(1.0, 1.5 * hull.length, 0.01),
    )
    bare_attitude = CraftBuildup.solve_equilibrium(**common)
    bare = CraftBuildup(
        sinkage=bare_attitude["sinkage"], trim=bare_attitude["trim"], **common
    ).run_with_stability()
    gml_bare = scalar(bare["GM_L"])
    print(f"{hull}, {mass:.1f} kg, CG {x_cg - float(hull.stations[0]):.2f} m aft of the bow")
    print(
        f"Bare hull at {args.velocity} m/s: D = {scalar(bare['D']):.1f} N, "
        f"GM_L = {gml_bare:.1f} m, trim {bare['trim']:+.2f} deg, "
        f"sinkage {1000 * bare['sinkage']:+.0f} mm\n"
    )

    def pair(text):
        return None if text is None else tuple(float(v) for v in text.split(","))

    freedoms = dict(
        cg_range=pair(args.cg_range),
        mass_range=pair(args.mass_range),
        thrust_height_range=pair(args.thrust_height_range),
    )
    any_freedom = any(v is not None for v in freedoms.values())

    floors: list[float | None] = [None] + [
        float(f) * gml_bare for f in args.gml_fractions.split(",") if f
    ]
    # Every (floor, starting layout) pair, and each floor's bare-hull baseline, is an
    # independent solve: run them all at once.
    tasks, labels = [], []
    for floor in floors:
        if any_freedom:
            tasks.append((hull, args.velocity, mass, xyz_cg, floor, None, False, water))
            labels.append((floor, "bare"))
        for start in starting_layouts(hull, args.stabilizer)[: args.starts]:
            tasks.append(
                (hull, args.velocity, mass, xyz_cg, floor, start, args.stabilizer, water)
            )
            labels.append((floor, "foils"))
    t0 = time.time()
    outcomes = hsb.parallel_map(
        optimize_task, [(task, freedoms) for task in tasks], jobs=args.jobs
    )
    print(f"{len(tasks)} optimizations in {time.time() - t0:.0f} s\n")

    for floor in floors:
        label = "No stability floor" if floor is None else f"GM_L >= {floor:.1f} m"
        print(f"--- {label} ---")
        mine = [o for o, (f, _) in zip(outcomes, labels) if f == floor]
        kinds = [k for (f, k) in labels if f == floor]
        for outcome, kind in zip(mine, kinds):
            if kind == "bare":
                if outcome is None:
                    print("Bare hull, loading optimized: did not converge")
                else:
                    print(describe("Bare hull, loading optimized", outcome, hull))
        candidates = [
            o
            for o, kind in zip(mine, kinds)
            if kind == "foils"
            and o is not None
            and (floor is None or o["results"]["GM_L"] >= 0.98 * floor)
        ]
        if not candidates:
            print("With foils: no feasible design found\n")
        else:
            best = min(candidates, key=lambda o: o["results"]["D"])
            print(describe("With foils", best, hull) + "\n")
