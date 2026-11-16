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
(``boatmath hull my.igs > hull.jsonl``), and ``--camber FILE`` a camber document
(see ``hydrosandbox.hull.camber``), with ``--mass`` and ``--lcg`` (metres aft of the
bow) describing its load. To optimize over a family of camber hulls, blending them,
see ``optimize_camber_blend.py``.
"""

import argparse
import json
import os
import time
import numpy as onp
import aerosandbox as asb
import aerosandbox.numpy as np
import hydrosandbox as hsb
from hydrosandbox.hull import CraftBuildup, Hull


def scalar(value) -> float:
    return float(onp.ravel(value)[0])


_SECTION_BASE = {}

# Second moment of area of a solid foil section about its chord line, as a fraction
# of chord * thickness^3 (from the NACA 4-digit thickness form).
SECTION_INERTIA_FACTOR = 0.0395


def foil_section(thickness: float, camber: float) -> asb.KulfanAirfoil:
    """
    A foil section of given maximum thickness and maximum camber (both as fractions
    of chord): the NACA 2412's thickness form and camber line, each scaled. Either
    may be an optimization variable.
    """
    if not _SECTION_BASE:
        base = asb.KulfanAirfoil("naca2412")
        x = onp.linspace(0.0, 1.0, 401)
        _SECTION_BASE.update(
            thickness_weights=0.5 * (base.upper_weights - base.lower_weights),
            camber_weights=0.5 * (base.upper_weights + base.lower_weights),
            leading_edge_weight=base.leading_edge_weight,
            TE_thickness=base.TE_thickness,
            thickness=float(onp.max(base.local_thickness(x))),
            camber=float(onp.max(base.local_camber(x))),
        )
    b = _SECTION_BASE
    kt, kc = thickness / b["thickness"], camber / b["camber"]
    return asb.KulfanAirfoil(
        name="section",
        upper_weights=kc * b["camber_weights"] + kt * b["thickness_weights"],
        lower_weights=kc * b["camber_weights"] - kt * b["thickness_weights"],
        leading_edge_weight=kc * b["leading_edge_weight"],
        TE_thickness=kt * b["TE_thickness"],
    )


def root_bending_stress(
    design: dict, layout: str, demihull_spacing: float | None, pressure: float
):
    """
    Bending stress [Pa] at the main foil's most loaded section, were it solid and
    loaded everywhere at ``pressure`` [Pa] (dynamic pressure times a limit lift
    coefficient), the load spread in proportion to chord.

    A foil on one strut is a cantilever each side of it. The hull-to-hull foil of
    ``layout="pi"`` is taken as simply supported on its two struts (they are too
    slender to clamp it), so its worst section is at midspan.
    """
    root_chord, taper = design["root_chord_main"], design["taper_main"]
    half_span = 0.5 * demihull_spacing if layout == "pi" else design["half_span_main"]
    half_load = pressure * half_span * 0.5 * (1.0 + taper) * root_chord
    centroid = half_span * (1.0 + 2.0 * taper) / (3.0 * (1.0 + taper))  # from the root
    moment = half_load * (half_span - centroid if layout == "pi" else centroid)
    thickness = design.get("thickness_main", 0.12) * root_chord
    return moment / (2.0 * SECTION_INERTIA_FACTOR * root_chord * thickness**2)


# ePropulsion Pod Drive 3.0 Evo: 3 kW input at 48 V, 17 kg, 570 N static thrust
# (manufacturer's parts database); its Navy 3.0 propeller, 10.2 x 6.7 in, two blades,
# to 2300 rpm. Lower-unit sizes are those of the Navy 3.0 Evo outboard, which shares
# it. The blade-area ratio and the leg's section are assumed.
EPROPULSION_POD_3 = dict(
    mass=17.0,
    propeller_diameter=0.259,
    propeller_pitch=0.170,
    blades=2,
    blade_area_ratio=0.30,
    max_rpm=2300.0,
    max_input_power=3000.0,
    motor_efficiency=0.88,
    body_length=0.30,
    body_diameter=0.101,
    leg_chord=0.117,
    leg_section="naca0030",
    # Least depth of water over the blade tips [m], against drawing air (the
    # hull's stern wave trough is some 0.07 m of it); and the least fore-aft gap
    # between a propeller and a foil strut in line with it, in diameters. Both
    # are judgement, standing in for ventilation, which is not modelled.
    min_tip_immersion=0.20,
    min_strut_gap=1.0,
)


def pod_components(pods: dict, x, depth, hull: Hull, demihull_spacing: float | None):
    """
    The drive pods as geometry: a body of revolution with its nose at ``x``, its axis
    ``depth`` below the waterline, and a leg from it up to the keel. One under each
    demihull of a catamaran, otherwise one on the centreline.
    """
    length, radius = pods["body_length"], 0.5 * pods["body_diameter"]
    shape = ((0.0, 0.0), (0.12, 0.75), (0.30, 1.0), (0.75, 1.0), (1.0, 0.55))
    sides = [0.0] if demihull_spacing is None else [0.5 * demihull_spacing, -0.5 * demihull_spacing]
    bodies = [
        hsb.HydroFuselage(
            name="Drive pod",
            xsecs=[
                asb.FuselageXSec(xyz_c=[x + f * length, y, -depth], radius=r * radius)
                for f, r in shape
            ],
        )
        for y in sides
    ]
    x_leg = x + 0.5 * length - 0.5 * pods["leg_chord"]
    leg_airfoil = asb.Airfoil(pods["leg_section"])
    leg = hsb.Strut(
        name="Pod legs",
        symmetric=demihull_spacing is not None,
        xsecs=[
            asb.WingXSec(xyz_le=[x_leg, sides[0], z], chord=pods["leg_chord"], airfoil=leg_airfoil)
            for z in (-depth + radius, hull.keel_height(x + 0.5 * length))
        ],
    )
    return bodies, leg


def propeller_components(pods: dict, x_disc: float, depth: float, demihull_spacing: float | None):
    """
    The propellers as geometry, for drawing only (not for analysis): each blade a
    twisted surface of constant pitch from hub to tip, with a hub fairing. Numeric
    inputs. Twin propellers are drawn counter-rotating, blades vertical.

    The blade outline and sections are generic (the maker's are not published):
    the area is the pods' blade-area ratio, the pitch their propeller's.
    """
    radius = 0.5 * pods["propeller_diameter"]
    hub = 0.2 * radius
    blades = pods["blades"]
    fractions = onp.array([0.2, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0])
    outline = onp.array([0.62, 0.80, 0.96, 1.00, 0.92, 0.70, 0.30])
    mean_chord = pods["blade_area_ratio"] * onp.pi * radius**2 / (blades * (radius - hub))
    span_fraction = (fractions - fractions[0]) / (fractions[-1] - fractions[0])
    chords = outline * mean_chord / onp.trapezoid(outline, span_fraction)
    sides = [(0.0, 1.0)] if demihull_spacing is None else [(0.5 * demihull_spacing, 1.0), (-0.5 * demihull_spacing, -1.0)]
    wings, hubs = [], []
    for y_disc, hand in sides:
        for k in range(blades):
            angle = 2.0 * onp.pi * k / blades
            radial = onp.array([0.0, onp.sin(angle), onp.cos(angle)])
            tangent = hand * onp.cross([1.0, 0.0, 0.0], radial)

            pitch_angles = onp.arctan2(pods["propeller_pitch"], 2.0 * onp.pi * fractions * radius)
            # Each section's chord runs downstream and back against the rotation.
            # The leading edge is a straight radial line (ahead of the blade's axis
            # by half the widest section), each section centred fore and aft.
            lead = 0.5 * onp.max(chords * onp.cos(pitch_angles))

            def blade(twist_sign):
                xsecs = []
                for f, chord, pitch_angle, s_f in zip(fractions, chords, pitch_angles, span_fraction):
                    at = (
                        onp.array([x_disc, y_disc, -depth]) + f * radius * radial + lead * tangent
                        - 0.5 * chord * onp.sin(pitch_angle) * onp.array([1.0, 0.0, 0.0])
                    )
                    xsecs.append(
                        asb.WingXSec(
                            xyz_le=list(at), chord=chord,
                            twist=twist_sign * (90.0 - onp.degrees(pitch_angle)),
                            airfoil=asb.Airfoil(f"naca24{int(round(10 - 6 * s_f)):02d}"),
                        )
                    )
                return hsb.Strut(name="Propeller blade", symmetric=False, xsecs=xsecs)

            def alignment(wing):
                """How well the mid-blade chord points where it should."""
                chord_direction = onp.array([float(v) for v in wing._compute_frame_of_WingXSec(3)[0]])
                wanted = onp.sin(pitch_angles[3]) * onp.array([1.0, 0.0, 0.0]) - onp.cos(pitch_angles[3]) * tangent
                return float(onp.dot(chord_direction, wanted))

            wings.append(max((blade(1.0), blade(-1.0)), key=alignment))
        hubs.append(
            hsb.HydroFuselage(
                name="Propeller hub",
                xsecs=[
                    asb.FuselageXSec(xyz_c=[x_disc + dx, y_disc, -depth], radius=r * hub)
                    for dx, r in ((-0.03, 1.0), (0.02, 1.0), (0.05, 0.7), (0.07, 0.15))
                ],
            )
        )
    return wings, hubs


def build_assembly(
    design: dict,
    stabilizer: bool,
    strut_freeboard: float = 0.25,
    layout: str = "centreline",
    hull: Hull | None = None,
    demihull_spacing: float | None = None,
    pods: dict | None = None,
    span_panels: int = 4,
    draw_propellers: bool = False,
) -> hsb.HydrofoilAssembly:
    """
    The foils, from design variables (none, if the design has no ``x_main``), and
    the drive pods if ``pods`` describes them (see ``pod_components``; the design
    then holds ``x_pod`` and ``depth_pod``). With pods, each panel of the lifting
    foils is cut into ``span_panels`` spanwise. ``draw_propellers`` adds the
    propellers' blades and hubs to a numeric design, for export only.
    """
    foils, struts = (
        _foil_components(design, stabilizer, strut_freeboard, layout, hull, demihull_spacing)
        if "x_main" in design
        else ([], [])
    )
    if pods is not None and span_panels > 1:
        # Finer spanwise sections on the lifting foils, so that the local flow of
        # a propeller reaches the part of the span that is near it.
        def between(a, b, t):
            return asb.WingXSec(
                xyz_le=[(1.0 - t) * a.xyz_le[k] + t * b.xyz_le[k] for k in range(3)],
                chord=(1.0 - t) * a.chord + t * b.chord,
                twist=(1.0 - t) * a.twist + t * b.twist,
                airfoil=a.airfoil,
            )

        for foil in foils:
            xsecs = [foil.xsecs[0]]
            for a, b in zip(foil.xsecs[:-1], foil.xsecs[1:]):
                xsecs += [between(a, b, k / span_panels) for k in range(1, span_panels)] + [b]
            foil.xsecs = xsecs
    bodies = []
    if pods is not None:
        bodies, leg = pod_components(
            pods, design["x_pod"], design["depth_pod"], hull, demihull_spacing
        )
        struts = struts + [leg]
        if draw_propellers:
            # (for drawing a solved design; not for analysis)
            blades, hubs = propeller_components(
                pods, design["x_pod"] + pods["body_length"], design["depth_pod"], demihull_spacing
            )
            struts, bodies = struts + blades, bodies + hubs
    return hsb.HydrofoilAssembly(
        name="Lift-assist foils", foils=foils, struts=struts, fuselages=bodies
    )


def _foil_components(
    design: dict,
    stabilizer: bool,
    strut_freeboard: float,
    layout: str,
    hull: Hull | None,
    demihull_spacing: float | None,
):
    """
    The foils and their struts.

    ``layout="centreline"``: a main T-foil on the craft's centreline, its strut
    running up through the waterline (and, optionally, a stabilizer T-foil).

    For a catamaran (``hull`` is a demihull, ``demihull_spacing`` apart):

    - ``layout="pi"``: one foil spanning from hull to hull, on two struts, one
      under each demihull. Its span is the hull spacing.
    - ``layout="twin"``: a T-foil under each demihull, each on its own strut.

    In both, each strut runs from the foil up to that hull's keel.
    """
    if "thickness_main" in design or "camber_main" in design:
        main_airfoil = foil_section(
            design.get("thickness_main", 0.12), design.get("camber_main", 0.02)
        )
    else:
        main_airfoil = asb.Airfoil("naca2412")
    strut_airfoil = asb.Airfoil("naca0012")

    if layout in ("pi", "twin"):
        x, depth = design["x_main"], design["depth_main"]
        root_chord = design["root_chord_main"]
        tip_chord = design["taper_main"] * root_chord
        incidence = design["incidence_main"]
        y_hull = 0.5 * demihull_spacing
        sweep = 0.25 * (root_chord - tip_chord)

        def section(x_le, y, chord):
            return asb.WingXSec(
                xyz_le=[x_le, y, -depth], chord=chord, twist=incidence, airfoil=main_airfoil
            )

        if layout == "pi":
            # Root chord on the centreline, tip chord where the foil meets each strut.
            foils = [
                hsb.Foil(
                    name="Main foil",
                    symmetric=True,
                    role="main",
                    xsecs=[section(x, 0.0, root_chord), section(x + sweep, y_hull, tip_chord)],
                )
            ]
            strut_chord, x_strut = 0.8 * tip_chord, x + sweep + 0.1 * tip_chord
        else:
            half_span = design["half_span_main"]
            foils = [
                hsb.Foil(
                    name=f"{side} foil",
                    symmetric=False,
                    role="main",
                    xsecs=[
                        section(x + sweep, y - half_span, tip_chord),
                        section(x, y, root_chord),
                        section(x + sweep, y + half_span, tip_chord),
                    ],
                )
                for side, y in (("Starboard", y_hull), ("Port", -y_hull))
            ]
            strut_chord, x_strut = 0.8 * root_chord, x + 0.1 * root_chord
        struts = [
            hsb.Strut(
                name="Hull struts",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[x_strut, y_hull, z], chord=strut_chord, airfoil=strut_airfoil
                    )
                    for z in (-depth, hull.keel_height(x_strut))
                ],
            )
        ]
        return foils, struts

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
            length=depth + strut_freeboard,
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
    return foils, struts


def design_bounds(
    hull: Hull, stabilizer: bool, layout: str = "centreline"
) -> dict[str, tuple[float, float]]:
    x_bow, x_stern = float(hull.stations[0]), float(hull.stations[-1])
    length = x_stern - x_bow
    if layout in ("pi", "twin"):
        bounds = {
            "x_main": (x_bow + 0.10 * length, x_bow + 0.95 * length),
            "depth_main": (0.25, 0.80),
            "root_chord_main": (0.05, 0.30),
            "taper_main": (0.40, 1.00),
            "incidence_main": (-3.0, 6.0),
        }
        if layout == "twin":
            bounds["half_span_main"] = (0.15, 0.50)
        return bounds
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
    layout: str = "centreline",
    keel_clearance: float = 0.10,
    objective: str = "drag",
    drag_cap: float | None = None,
    damping_floor: float | None = None,
    use_hull_tables: bool = True,
    optimize_section: bool = False,
    allowable_stress: float | None = None,
    structural_lift_coefficient: float = 1.0,
    pods: dict | None = None,
    pod_start: tuple[float, float] | None = None,
    propeller_interaction: bool = True,
    blend_start: list[float] | None = None,
    transform_start: dict | None = None,
    hull_constraints=None,
    cg_level_at_rest: bool = False,
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
    ``design_bounds``. ``layout`` chooses the arrangement (see ``build_assembly``);
    under-hull layouts keep the foil ``keel_clearance`` [m] below the keel above it.

    ``objective`` is what to optimize: ``"drag"`` (minimize it), ``"pitch_stiffness"``
    (maximize the heave-free pitch stiffness, as ``GM_L``) or ``"pitch_damping"``
    (maximize the foils' pitch damping). ``drag_cap`` [N], ``gml_floor`` [m] and
    ``damping_floor`` [N m s/rad] constrain the others. Pass ``use_hull_tables=False``
    to disable automatic attitude tabulation of the hull's dynamic force and wave
    amplitude.

    ``optimize_section=True`` adds the main foil's section thickness and camber
    (``thickness_main``, ``camber_main``, fractions of chord; see ``foil_section``)
    to the design variables. ``allowable_stress`` [Pa] then limits the bending stress
    of the main foil, as a solid section loaded at ``structural_lift_coefficient``
    (see ``root_bending_stress``); without it, nothing rewards thickness.

    ``pods`` (e.g. ``EPROPULSION_POD_3``) drives the craft with pod drives: one under
    each demihull, or one on the centreline. Their fore-aft position and depth
    (``x_pod``, ``depth_pod``) and shaft speed (``rpm``) become variables;
    ``pod_start`` gives the first two to start from. Their mass is part of ``mass``
    and sits at the pods; the thrust acts there; their bodies and legs add drag and
    waves; and the propellers (Wageningen B-series, open water: no wake or thrust
    deduction) must give the thrust within the motors' power and speed.

    With ``propeller_interaction`` the propellers and foils also act on each other
    (``hydrosandbox.propulsion.interaction``): each propeller's inflow, jet and swirl
    reach the foils and struts as a change in their local flow (twin propellers
    counter-rotating), and the foils' viscous wakes and the flow their lift induces
    reach the propellers as a wake fraction and an inflow angle. The blade tips are
    kept ``min_tip_immersion`` under the still surface, and a propeller in line
    with a foil strut ``min_strut_gap`` diameters clear of it fore and aft.

    ``hull`` may also be a camber ``HullFamily`` (``hydrosandbox.hull.camber``): its
    blend weights then become design variables too (a slider ``blend_t`` for two
    hulls, else ``blend_w0, blend_w1, ...`` on the simplex), starting from
    ``blend_start`` (default: the family's centroid). The hull inside the solve is
    the family's polynomial interpolant; the solution is re-evaluated on the exact
    blend, swept from camber's blended control points, which is returned as
    ``"hull"`` with the weights as ``"blend_weights"``.

    ``hull`` may likewise be a ``DesignFamily`` (``hydrosandbox.hull.transformed``:
    e.g. a ``TransformedFamily`` of one hull's beam, draft, Lackenby prismatic and
    LCB, bow rake, or an ``FFDFamily``). Each of its variables becomes a variable
    ``hull_<name>`` within the family's range, starting from ``transform_start``
    (default: the family's ``start``).
    The solution is re-evaluated on the exact transformed hull, returned as
    ``"hull"`` with the values as ``"transform"``.

    ``hull_constraints(hull)`` (optional) returns further constraints on the hull
    inside the solve (e.g. draft or freeboard limits on a family's interpolant).

    ``cg_level_at_rest=True`` puts the centre of gravity (fore and aft) wherever the
    hull floats level at rest under ``mass``: at its centre of buoyancy at the
    level sinkage that displaces it. The CG then follows the hull when the hull is a
    variable (a blend), which ``xyz_cg[0]`` and ``cg_range`` are ignored for.
    """
    from hydrosandbox.hull.camber import HullFamily
    from hydrosandbox.hull.transformed import DesignFamily

    family = hull if isinstance(hull, HullFamily) else None
    transform = hull if isinstance(hull, DesignFamily) else None
    if transform is not None:
        transform_start = {
            k: float(onp.clip((transform_start or {}).get(k, transform.start[k]), *transform.ranges[k]))
            for k in transform.names
        }
        hull = transform.at(transform_start)
    if family is not None:
        blend_start = list(blend_start) if blend_start is not None else [1.0 / family.n_hulls] * family.n_hulls
        hull = family.at(blend_start)
    n_hulls = 1 if (craft_options or {}).get("demihull_spacing") is None else 2

    def level_at_rest(h, m):
        """Level sinkage at which hull(s) `h` displace mass `m`, and the LCB there."""
        from scipy.optimize import brentq

        volume = m / water.density() / n_hulls
        s0 = brentq(
            lambda s_: float(h.hydrostatics(s_, 0.0, (0.0, 0.0))["volume"]) - volume, -0.5, 0.5, xtol=1e-7
        )
        return s0, float(h.hydrostatics(s0, 0.0, (0.0, 0.0))["x_cb"])

    if cg_level_at_rest:
        rest_start, x_cg_start = level_at_rest(hull, mass)
        xyz_cg = [x_cg_start, xyz_cg[1], xyz_cg[2]]
        cg_range = None
    with_foils = start is not None
    with_assembly = with_foils or pods is not None
    n_pods = 0 if pods is None else (1 if (craft_options or {}).get("demihull_spacing") is None else 2)
    if pods is not None:
        if thrust_height_range is not None:
            raise ValueError("With pods the thrust acts at the pods; omit thrust_height_range.")
        x_pod_start, depth_pod_start = pod_start or (
            float(hull.stations[0]) + 0.85 * hull.length,
            0.35,
        )
        start = dict(start or {}, x_pod=x_pod_start, depth_pod=depth_pod_start)
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
    exact_common = {
        k: v
        for k, v in common.items()
        if k not in ("hull_dynamics", "hull_wave_amplitude")
    }
    if use_hull_tables:
        table_trim = (min(float(trim_bounds[0]), -2.0), max(float(trim_bounds[1]), 4.0))
        if common.get("include_hull_dynamics", True) and "hull_dynamics" not in common:
            common["hull_dynamics"] = hull.dynamic_force_table(
                velocity,
                water.density(),
                water.gravity,
                sinkage_range=(-0.4, 0.2),
                trim_range=table_trim,
                n_sinkage=11,
                n_trim=11,
                demihull_spacing=common.get("demihull_spacing"),
                jobs=1,
            )
        if "hull_wave_amplitude" not in common:
            common["hull_wave_amplitude"] = hull.wave_amplitude_table(
                velocity,
                water.gravity,
                length_scales=common["wave_length_scales"],
                sinkage_range=(-0.4, 0.2),
                trim_range=table_trim,
                n_sinkage=11,
                n_trim=11,
                demihull_spacing=common.get("demihull_spacing"),
                jobs=1,
            )
    layout_options = dict(
        layout=layout, hull=hull, demihull_spacing=common.get("demihull_spacing"), pods=pods
    )

    def craft_inputs(values: dict, placement: dict) -> dict:
        """Mass, centre of gravity and thrust point for a loading and a design."""
        if pods is None:
            return dict(
                mass=values["mass"],
                xyz_cg=[values["x_cg"], 0.0, xyz_cg[2]],
                xyz_thrust=(
                    None
                    if values["thrust_height"] is None
                    else [x_stern, 0.0, values["thrust_height"]]
                ),
            )
        at_pods = [placement["x_pod"] + 0.5 * pods["body_length"], 0.0, -placement["depth_pod"]]
        return dict(
            mass=values["mass"] - n_pods * pods["mass"],
            xyz_cg=[values["x_cg"], 0.0, xyz_cg[2]],
            point_masses=[(n_pods * pods["mass"], at_pods)],
            xyz_thrust=[placement["x_pod"] + pods["body_length"], 0.0, -placement["depth_pod"]],
        )

    try:
        at_start = CraftBuildup.solve_equilibrium(
            assembly=build_assembly(start, stabilizer, **layout_options) if with_assembly else None,
            **craft_inputs(
                {"mass": mass, "x_cg": xyz_cg[0], "thrust_height": None if thrust_start is None else thrust_start[2]},
                start or {},
            ),
            **common,
        )
    except RuntimeError:
        return None

    opti = asb.Opti()
    design = {}
    if with_foils:
        design = {
            name: opti.variable(init_guess=start[name], lower_bound=lo, upper_bound=hi)
            for name, (lo, hi) in {
                **design_bounds(hull, stabilizer, layout), **(bounds or {})
            }.items()
            if name in design_bounds(hull, stabilizer, layout)
        }
        if optimize_section:
            for name, default, limits in (
                ("thickness_main", 0.12, (0.06, 0.18)),
                ("camber_main", 0.02, (0.0, 0.06)),
            ):
                lo, hi = (bounds or {}).get(name, limits)
                design[name] = opti.variable(
                    init_guess=start.get(name, default), lower_bound=lo, upper_bound=hi
                )
    rpm = None
    if pods is not None:
        x_bow, radius = float(hull.stations[0]), 0.5 * pods["propeller_diameter"]
        pod_bounds = {
            "x_pod": (x_bow + 0.30 * hull.length, x_stern - pods["body_length"]),
            "depth_pod": (radius + pods.get("min_tip_immersion", 0.10), 0.60),
        }
        for name, limits in pod_bounds.items():
            lo, hi = (bounds or {}).get(name, limits)
            design[name] = opti.variable(init_guess=start[name], lower_bound=lo, upper_bound=hi)
        rpm = pods["max_rpm"] * opti.variable(init_guess=0.92, lower_bound=0.2, upper_bound=1.0)
    blend_constraints = []
    if family is not None:
        # The blend: the hull inside the solve becomes the family's interpolant at
        # variable weights, and so does everything built from it.
        if family.n_hulls == 2:
            design["blend_t"] = opti.variable(init_guess=blend_start[1], lower_bound=0.0, upper_bound=1.0)
            blend_weights = [1.0 - design["blend_t"], design["blend_t"]]
        else:
            free = [
                opti.variable(init_guess=blend_start[i], lower_bound=0.0, upper_bound=1.0)
                for i in range(family.n_hulls - 1)
            ]
            design.update({f"blend_w{i}": w for i, w in enumerate(free)})
            blend_weights = free + [1.0 - sum(free)]
            blend_constraints.append(blend_weights[-1] >= 0.0)
        hull = family.at(blend_weights)
        common["hull"] = hull
        for key in ("hull_dynamics", "hull_wave_amplitude"):
            if key in common:
                common[key] = hull.rebind(common[key])
        layout_options["hull"] = hull
    if transform is not None:
        # The transformations: each a variable within its range; the hull inside the
        # solve becomes the family's interpolant there.
        transform_values = {}
        for k in transform.names:
            lo, hi = transform.ranges[k]
            design[f"hull_{k}"] = transform_values[k] = opti.variable(
                init_guess=transform_start[k], lower_bound=lo, upper_bound=hi, scale=0.5 * (hi - lo)
            )
        hull = transform.at(transform_values)
        common["hull"] = hull
        for key in ("hull_dynamics", "hull_wave_amplitude"):
            if key in common:
                common[key] = hull.rebind(common[key])
        layout_options["hull"] = hull
    interacting = pods is not None and propeller_interaction
    wake_fraction = inflow_angle = 0.0
    if interacting:
        # What the foils do to the propellers' inflow, as variables tied below to
        # what the solved flow implies (the two depend on each other).
        wake_fraction = 0.1 * opti.variable(init_guess=0.0, lower_bound=-3.0, upper_bound=5.0)
        inflow_angle = opti.variable(
            init_guess=float(at_start["trim"]), lower_bound=-15.0, upper_bound=15.0
        )
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

    rest_constraints = []
    if cg_level_at_rest:
        rest_sinkage = opti.variable(init_guess=rest_start, lower_bound=-0.5, upper_bound=0.5)
        at_rest = hull.hydrostatics(rest_sinkage, 0.0, (0.0, 0.0))
        loading["x_cg"] = at_rest["x_cb"]
        rest_constraints.append(
            n_hulls * at_rest["volume"] * water.density() / loading["mass"] == 1.0
        )
    sinkage = opti.variable(init_guess=at_start["sinkage"], lower_bound=-0.4, upper_bound=0.2)
    trim = opti.variable(
        init_guess=float(onp.clip(at_start["trim"], *trim_bounds)),
        lower_bound=trim_bounds[0],
        upper_bound=trim_bounds[1],
    )

    def propellers(shaft_rpm, depth, wake=0.0, angle=0.0):
        from hydrosandbox.propulsion.bseries import evaluate_bseries

        return evaluate_bseries(
            shaft_rpm,
            pods["propeller_diameter"],
            pods["propeller_pitch"] / pods["propeller_diameter"],
            pods["blade_area_ratio"],
            n_blades=pods["blades"],
            v_ship=velocity,
            wake_fraction=wake,
            shaft_angle_deg=angle,
            depth=depth,
            water=water,
        )

    spacing = common.get("demihull_spacing")
    disc_sides = [0.0] if spacing is None else [0.5 * spacing, -0.5 * spacing]

    def disc_centre(placement, y=None):
        return [
            placement["x_pod"] + pods["body_length"],
            disc_sides[0] if y is None else y,
            -placement["depth_pod"],
        ]

    def propeller_inflow(placement, at):
        """The propellers' induced flow at a point, as ``CraftBuildup`` takes it."""
        from hydrosandbox.propulsion.interaction import actuator_disc_velocity

        def flow(x, y, z):
            total = [0.0, 0.0, 0.0]
            for y_disc in disc_sides:
                part = actuator_disc_velocity(
                    x, y, z, disc_centre(placement, y_disc), 0.5 * pods["propeller_diameter"],
                    at["thrust"], velocity, water.density(), torque=at["torque"],
                    rotation=1.0 if y_disc >= 0 else -1.0,
                )
                total = [total[k] + part[k] for k in range(3)]
            return total

        return flow

    def at_propeller(craft_assembly, foil_results, placement):
        """The foils' effect on a propeller's inflow (the pods' own legs excluded)."""
        from hydrosandbox.propulsion.interaction import foils_at_propeller

        pairs = [
            (wing, component)
            for wing, component in zip(craft_assembly.wings, foil_results["wing_hydro_components"])
            if wing.name != "Pod legs"
        ]
        return foils_at_propeller(
            [w for w, _ in pairs], [c for _, c in pairs], disc_centre(placement),
            0.5 * pods["propeller_diameter"], velocity, water.density(),
        )

    propeller = None
    if pods is not None:
        propeller = propellers(rpm, design["depth_pod"], wake_fraction, inflow_angle)
    assembly = build_assembly(design, stabilizer, **layout_options) if with_assembly else None
    craft = CraftBuildup(
        assembly=assembly, sinkage=sinkage, trim=trim, **craft_inputs(loading, design), **common,
        inflow_perturbation=propeller_inflow(design, propeller) if interacting else None,
    )
    needs_damping = objective == "pitch_damping" or damping_floor is not None
    needs_stability = objective != "drag" or gml_floor is not None or needs_damping
    res = (
        craft.run_with_stability(include_pitch_damping=needs_damping)
        if needs_stability
        else craft.run()
    )
    weight = mass * water.gravity
    q = 0.5 * water.density() * velocity**2

    constraints = [
        res["F_z_net"] / weight == 0,
        res["M_y_net"] / (weight * hull.length) == 0,
    ] + blend_constraints + rest_constraints
    if hull_constraints is not None:
        constraints += list(hull_constraints(hull))
    if pods is not None:
        shaft_power_limit = pods["max_input_power"] * pods["motor_efficiency"]
        constraints += [
            n_pods * propeller["thrust"] / (0.1 * weight) == res["thrust"] / (0.1 * weight),
            propeller["P_shaft"] / shaft_power_limit <= 1.0,
            # The propeller turns clear below the hull.
            design["depth_pod"]
            >= -hull.keel_height(design["x_pod"] + pods["body_length"])
            + 0.5 * pods["propeller_diameter"]
            + 0.02,
        ]
        if interacting:
            seen = at_propeller(assembly, res["foil"], design)
            constraints += [
                wake_fraction / 0.1 == seen["wake_fraction"] / 0.1,
                inflow_angle == trim + seen["inflow_angle"],
            ]
        if with_foils and layout in ("pi", "twin"):
            # A foil strut stands in line with each propeller: keep them apart.
            root, tip = design["root_chord_main"], design["taper_main"] * design["root_chord_main"]
            strut_chord = 0.8 * (tip if layout == "pi" else root)
            x_strut = design["x_main"] + (
                0.25 * (root - tip) + 0.5 * tip if layout == "pi" else 0.5 * root
            )
            gap = pods.get("min_strut_gap", 1.0) * pods["propeller_diameter"] + 0.5 * strut_chord
            constraints.append(((disc_centre(design)[0] - x_strut) / gap) ** 2 >= 1.0)
    if with_foils:
        foil = res["foil"]
        constraints += [
            foil["cavitation_margin"] >= 0.3,
            foil["ventilation_margin"] >= 0.1,
        ]
        if layout != "pi":
            # A cantilevered foil: aspect ratio within a sensible structural limit.
            constraints.append(
                2.0 * design["half_span_main"]
                <= 10.0 * 0.5 * (1.0 + design["taper_main"]) * design["root_chord_main"]
            )
        if layout in ("pi", "twin"):
            # The foil stays clear below the hull above it.
            constraints.append(
                design["depth_main"] >= -hull.keel_height(design["x_main"]) + keel_clearance
            )
        # Foil loading within stall limits.
        for wing, component in zip(assembly.wings, foil["wing_hydro_components"]):
            if not getattr(wing, "is_strut", False):
                lift_coefficient = component.L / (q * wing.area())
                constraints += [lift_coefficient <= 0.7, lift_coefficient >= -0.5]
        if stabilizer:
            constraints.append(design["x_stab"] >= design["x_main"] + 0.5)
        if allowable_stress is not None:
            constraints.append(
                root_bending_stress(
                    design,
                    layout,
                    common.get("demihull_spacing"),
                    structural_lift_coefficient * q,
                )
                / allowable_stress
                <= 1.0
            )
    if gml_floor is not None:
        # (scaled so a zero or negative floor is also well-posed)
        gml_scale = max(abs(float(gml_floor)), 1.0)
        constraints.append((res["GM_L"] - gml_floor) / gml_scale >= 0.0)
    if damping_floor is not None:
        constraints.append(res["c_pitch_foils"] / damping_floor >= 1.0)
    if drag_cap is not None:
        constraints.append(res["D"] / drag_cap <= 1.0)

    opti.subject_to(constraints)
    if objective == "drag":
        opti.minimize(res["D"] / (0.1 * weight))
    elif objective == "pitch_stiffness":
        opti.minimize(-res["GM_L"] / hull.length)
    elif objective == "pitch_damping":
        opti.minimize(-res["c_pitch_foils"] / (weight * hull.length**2 / velocity))
    else:
        raise ValueError(f"Unknown objective {objective!r}.")
    try:
        sol = opti.solve(
            verbose=bool(os.environ.get("OPTIMIZE_VERBOSE")),
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
    solved_weights = None
    if family is not None:
        # Re-evaluate on the exact blend, not the interpolant.
        solved_weights = [float(onp.clip(sol(w), 0.0, 1.0)) for w in blend_weights]
        solved_weights = [w / sum(solved_weights) for w in solved_weights]
        hull = family.exact(solved_weights)
        exact_common = dict(exact_common, hull=hull)
        if cg_level_at_rest:
            solved_loading["x_cg"] = level_at_rest(hull, solved_loading["mass"])[1]
        layout_options["hull"] = hull
    solved_transform = None
    if transform is not None:
        # Re-evaluate on the exact transformed hull, not the interpolant.
        solved_transform = {
            k: float(onp.clip(solved[f"hull_{k}"], *transform.ranges[k])) for k in transform.names
        }
        hull = transform.exact(solved_transform)
        exact_common = dict(exact_common, hull=hull)
        if cg_level_at_rest:
            solved_loading["x_cg"] = level_at_rest(hull, solved_loading["mass"])[1]
        layout_options["hull"] = hull
    # Re-evaluate numerically at the solution, with the exact direct integrals and
    # interference breakdown.
    final = CraftBuildup(
        assembly=build_assembly(solved, stabilizer, **layout_options) if with_assembly else None,
        sinkage=float(sol(sinkage)),
        trim=float(sol(trim)),
        **craft_inputs(solved_loading, solved),
        **dict(exact_common, include_interference_breakdown=with_assembly),
        inflow_perturbation=(
            propeller_inflow(
                solved,
                propellers(float(sol(rpm)), solved["depth_pod"], float(sol(wake_fraction)), float(sol(inflow_angle))),
            )
            if interacting
            else None
        ),
    ).run_with_stability()
    seen = None
    if interacting:
        seen = at_propeller(
            build_assembly(solved, stabilizer, **layout_options), final["foil"], solved
        )
    x_cg_overall = scalar(final["xyz_cg"][0])
    # Keep the returned results plain (numbers only), so they can cross processes.
    final = {
        key: (scalar(value) if key not in ("sinkage", "trim") else float(value))
        for key, value in final.items()
        if key not in ("foil", "transom", "xyz_cg") and value is not None
    }
    if with_foils:
        final["root_bending_stress"] = float(
            root_bending_stress(
                solved,
                layout,
                common.get("demihull_spacing"),
                structural_lift_coefficient * 0.5 * water.density() * velocity**2,
            )
        )
    final["x_cg_overall"] = x_cg_overall
    if pods is not None:
        wake_solved = scalar(seen["wake_fraction"]) if interacting else 0.0
        angle_solved = float(sol(trim)) + scalar(seen["inflow_angle"]) if interacting else 0.0
        at = propellers(float(sol(rpm)), solved["depth_pod"], wake_solved, angle_solved)
        area = onp.pi * (0.5 * pods["propeller_diameter"]) ** 2
        loading_coefficient = scalar(at["thrust"]) / (0.5 * water.density() * velocity**2 * area)
        final.update(
            rpm=float(sol(rpm)),
            pods=n_pods,
            propeller_thrust_each=scalar(at["thrust"]),
            shaft_power_each=scalar(at["P_shaft"]),
            input_power_each=scalar(at["P_shaft"]) / pods["motor_efficiency"],
            propeller_efficiency=scalar(at["eta_0"]),
            propeller_jet_speedup=float(onp.sqrt(1.0 + loading_coefficient) - 1.0),
        )
        if interacting:
            final.update(
                wake_fraction=wake_solved,
                wake_fraction_viscous=scalar(seen["wake_fraction_viscous"]),
                propeller_inflow_angle=angle_solved,
            )
    outcome = {"design": solved, "loading": solved_loading, "results": final}
    # What the solve itself predicted (on a family's interpolant, before the exact
    # re-evaluation above), for trust-region steps.
    outcome["predicted"] = {
        key: float(sol(res[key])) for key in ("D", "GM_L") if key in res and res[key] is not None
    }
    if family is not None:
        outcome.update(blend_weights=solved_weights, hull=hull)
    if transform is not None:
        outcome.update(transform=solved_transform, hull=hull)
    return outcome


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
    parser.add_argument("--camber", type=str, default=None, help="camber hull document (JSON)")
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
    if args.camber:
        hull = Hull.from_camber(args.camber, n_stations=61, n_levels=41)
    elif args.hull_record:
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

    wave_scales = (1.0, 1.5 * hull.length, 0.01)
    shared_options = dict(
        hull_dynamics=hull.dynamic_force_table(
            args.velocity, water.density(), water.gravity,
            sinkage_range=(-0.4, 0.2), trim_range=(-2.0, 4.0),
            n_sinkage=11, n_trim=11, jobs=args.jobs,
        ),
        hull_wave_amplitude=hull.wave_amplitude_table(
            args.velocity, water.gravity, length_scales=wave_scales,
            sinkage_range=(-0.4, 0.2), trim_range=(-2.0, 4.0),
            n_sinkage=11, n_trim=11, jobs=args.jobs,
        ),
    )

    freedoms = dict(
        cg_range=pair(args.cg_range),
        mass_range=pair(args.mass_range),
        thrust_height_range=pair(args.thrust_height_range),
        craft_options=shared_options,
    )
    any_freedom = any(
        v is not None for k, v in freedoms.items() if k != "craft_options"
    )

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
    task_labels = [
        ("no floor" if floor is None else f"GM_L >= {floor:.1f} m")
        + (", bare hull" if kind == "bare" else f", foils, start {i}")
        for i, (floor, kind) in enumerate(labels)
    ]
    outcomes = hsb.parallel_map(
        optimize_task,
        [(task, freedoms) for task in tasks],
        jobs=args.jobs,
        progress=True,
        labels=task_labels,
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
