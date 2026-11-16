"""
Supplied design spaces: common foil arrangements, with default bounds and starts.

Each function returns a ``DesignSpace`` whose variables can be edited before use.
All positions are in the boat's axes (x aft, z up from the design waterline).
"""

from __future__ import annotations

from typing import Any

import numpy as onp

import aerosandbox as asb
from hydrosandbox.design.boat import Boat
from hydrosandbox.design.design_space import DesignSpace
from hydrosandbox.design.objective import AtLeast, AtMost
from hydrosandbox.geometry import Foil, HydrofoilAssembly, Strut

#: Second moment of area of a solid foil section about its chord line, as a fraction
#: of chord * thickness^3 (from the NACA 4-digit thickness form).
SECTION_INERTIA_FACTOR = 0.0395

_SECTION_BASE: dict[str, Any] = {}


def foil_section(thickness, camber) -> asb.KulfanAirfoil:
    """
    A foil section of given maximum thickness and camber (fractions of chord): the
    NACA 2412's thickness form and camber line, each scaled. Either may be symbolic.
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


def _section(values):
    if "thickness" in values or "camber" in values:
        return foil_section(values.get("thickness", 0.12), values.get("camber", 0.02))
    return asb.Airfoil("naca2412")


def _bending_stress(values, half_span, supported_at_tips: bool, pressure: float):
    """
    Bending stress [Pa] at the foil's most loaded section, were it solid and loaded
    everywhere at ``pressure`` [Pa], the load spread in proportion to chord. A foil
    held at its tips is taken as simply supported there (worst at midspan); one on
    a single strut is a cantilever each side of it.
    """
    root_chord, taper = values["root_chord"], values["taper"]
    half_load = pressure * half_span * 0.5 * (1.0 + taper) * root_chord
    centroid = half_span * (1.0 + 2.0 * taper) / (3.0 * (1.0 + taper))
    moment = half_load * (half_span - centroid if supported_at_tips else centroid)
    thickness = values.get("thickness", 0.12) * root_chord
    return moment / (2.0 * SECTION_INERTIA_FACTOR * root_chord * thickness**2)


def _common(boat: Boat, section: bool, start_fraction: float):
    hull = boat.hull
    x_bow, length = float(hull.stations[0]), hull.length
    variables = {
        "x": (x_bow + 0.10 * length, x_bow + 0.95 * length),
        "depth": (0.20, 0.80),
        "root_chord": (0.05, 0.30),
        "taper": (0.40, 1.00),
        "incidence": (-3.0, 6.0),
    }
    start = {"x": x_bow + start_fraction * length, "depth": 0.45, "root_chord": 0.14, "taper": 0.7, "incidence": 0.5}
    if section:
        variables.update(thickness=(0.06, 0.18), camber=(0.0, 0.06))
        start.update(thickness=0.12, camber=0.02)
    return variables, start


def two_strut_foil(boat: Boat, section: bool = False, start_fraction: float = 0.85,
                   structural_lift_coefficient: float = 1.0, velocity: float | None = None) -> DesignSpace:
    """
    One foil spanning from hull to hull of a catamaran, on a strut under each
    demihull. Variables: fore-aft position ``x``, ``depth``, ``root_chord`` (on the
    centreline), ``taper`` (chord at the struts over root chord), ``incidence``
    [deg], and with ``section=True`` the section's ``thickness`` and ``camber``.

    Quantities: ``keel_clearance`` [m] (foil below the keel above it) and, if
    ``velocity`` is given, ``bending_stress`` [Pa] at ``structural_lift_coefficient``.
    """
    if boat.demihull_spacing is None:
        raise ValueError("two_strut_foil needs a catamaran (Boat.demihull_spacing).")
    y_hull = 0.5 * boat.demihull_spacing
    strut_airfoil = asb.Airfoil("naca0012")

    def builder(values, boat):
        x, depth, root = values["x"], values["depth"], values["root_chord"]
        tip = values["taper"] * root
        sweep = 0.25 * (root - tip)
        airfoil = _section(values)
        foil = Foil(
            name="Main foil", symmetric=True, role="main",
            xsecs=[
                asb.WingXSec(xyz_le=[x, 0.0, -depth], chord=root, twist=values["incidence"], airfoil=airfoil),
                asb.WingXSec(xyz_le=[x + sweep, y_hull, -depth], chord=tip, twist=values["incidence"], airfoil=airfoil),
            ],
        )
        x_strut = x + sweep + 0.1 * tip
        struts = Strut(
            name="Hull struts", symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[x_strut, y_hull, z], chord=0.8 * tip, airfoil=strut_airfoil)
                for z in (-depth, boat.hull.keel_height(x_strut))
            ],
        )
        return HydrofoilAssembly(name="Two-strut foil", foils=[foil], struts=[struts])

    variables, start = _common(boat, section, start_fraction)
    quantities = {"keel_clearance": lambda v, b: v["depth"] + b.hull.keel_height(v["x"])}
    if velocity is not None:
        pressure = structural_lift_coefficient * 0.5 * boat.water.density() * velocity**2
        quantities["bending_stress"] = lambda v, b: _bending_stress(v, y_hull, True, pressure)
    return DesignSpace(builder, variables, start, quantities)


def _t_foil(name, x, y, depth, half_span, root, tip, incidence, airfoil, symmetric):
    sweep = 0.25 * (root - tip)

    def xsec(x_le, y_le, chord):
        return asb.WingXSec(xyz_le=[x_le, y_le, -depth], chord=chord, twist=incidence, airfoil=airfoil)

    if symmetric:
        xsecs = [xsec(x, 0.0, root), xsec(x + sweep, half_span, tip)]
    else:
        xsecs = [xsec(x + sweep, y - half_span, tip), xsec(x, y, root), xsec(x + sweep, y + half_span, tip)]
    return Foil(name=name, symmetric=symmetric, role="main", xsecs=xsecs)


def centreline_t_foil(boat: Boat, section: bool = False, start_fraction: float = 0.85,
                      strut_freeboard: float = 0.25, structural_lift_coefficient: float = 1.0,
                      velocity: float | None = None) -> DesignSpace:
    """
    A T-foil on the craft's centreline, its strut running up through the waterline.
    Variables as ``two_strut_foil``, plus ``half_span``. Quantities:
    ``aspect_ratio`` and, if ``velocity`` is given, ``bending_stress``.
    """
    strut_airfoil = asb.Airfoil("naca0012")

    def builder(values, boat):
        root = values["root_chord"]
        foil = _t_foil("Main foil", values["x"], 0.0, values["depth"], values["half_span"], root,
                       values["taper"] * root, values["incidence"], _section(values), True)
        strut = Strut.from_dimensions(
            name="Main foil strut", root_xyz=[values["x"] + 0.1 * root, 0.0, -values["depth"]],
            length=values["depth"] + strut_freeboard, root_chord=0.8 * root, tip_chord=0.8 * root,
            airfoil=strut_airfoil,
        )
        return HydrofoilAssembly(name="Centreline T-foil", foils=[foil], struts=[strut])

    variables, start = _common(boat, section, start_fraction)
    variables["half_span"], start["half_span"] = (0.20, 0.85), 0.55
    quantities = {
        "aspect_ratio": lambda v, b: 2.0 * v["half_span"] / (0.5 * (1.0 + v["taper"]) * v["root_chord"]),
    }
    if velocity is not None:
        pressure = structural_lift_coefficient * 0.5 * boat.water.density() * velocity**2
        quantities["bending_stress"] = lambda v, b: _bending_stress(v, v["half_span"], False, pressure)
    return DesignSpace(builder, variables, start, quantities)


def twin_t_foils(boat: Boat, section: bool = False, start_fraction: float = 0.85,
                 structural_lift_coefficient: float = 1.0, velocity: float | None = None) -> DesignSpace:
    """
    A T-foil under each demihull of a catamaran, each on a strut up to that hull's
    keel. Variables as ``centreline_t_foil``. Quantities: ``aspect_ratio``,
    ``keel_clearance`` and, if ``velocity`` is given, ``bending_stress``.
    """
    if boat.demihull_spacing is None:
        raise ValueError("twin_t_foils needs a catamaran (Boat.demihull_spacing).")
    y_hull = 0.5 * boat.demihull_spacing
    strut_airfoil = asb.Airfoil("naca0012")

    def builder(values, boat):
        root = values["root_chord"]
        airfoil = _section(values)
        foils = [
            _t_foil(f"{side} foil", values["x"], y, values["depth"], values["half_span"], root,
                    values["taper"] * root, values["incidence"], airfoil, False)
            for side, y in (("Starboard", y_hull), ("Port", -y_hull))
        ]
        x_strut = values["x"] + 0.1 * root
        struts = Strut(
            name="Hull struts", symmetric=True,
            xsecs=[
                asb.WingXSec(xyz_le=[x_strut, y_hull, z], chord=0.8 * root, airfoil=strut_airfoil)
                for z in (-values["depth"], boat.hull.keel_height(x_strut))
            ],
        )
        return HydrofoilAssembly(name="Twin T-foils", foils=foils, struts=[struts])

    variables, start = _common(boat, section, start_fraction)
    variables["half_span"], start["half_span"] = (0.15, 0.50), 0.35
    quantities = {
        "aspect_ratio": lambda v, b: 2.0 * v["half_span"] / (0.5 * (1.0 + v["taper"]) * v["root_chord"]),
        "keel_clearance": lambda v, b: v["depth"] + b.hull.keel_height(v["x"]),
    }
    if velocity is not None:
        pressure = structural_lift_coefficient * 0.5 * boat.water.density() * velocity**2
        quantities["bending_stress"] = lambda v, b: _bending_stress(v, v["half_span"], False, pressure)
    return DesignSpace(builder, variables, start, quantities)


def no_appendages() -> DesignSpace:
    """The bare boat: no foils, no design variables of its own."""
    return DesignSpace(lambda values, boat: None, {}, {})


def standard_penalties(design: DesignSpace, weight: float = 1.0) -> list:
    """
    The limits that the earlier optimizer had built in, as penalty terms for an
    ``Objective``: foil cavitation and ventilation margins, foil loading within
    stall, and, where the design space reports them, aspect ratio and keel
    clearance. Edit the list, or leave terms out, as the study needs.
    """
    terms = [
        AtLeast("cavitation_margin", 0.3, weight=weight, scale=0.1),
        AtLeast("ventilation_margin", 0.1, weight=weight, scale=0.05),
        AtMost("lift_coefficient_max", 0.7, weight=weight),
        AtLeast("lift_coefficient_min", -0.5, weight=weight, scale=0.05),
    ]
    if "aspect_ratio" in design.quantities:
        terms.append(AtMost("aspect_ratio", 10.0, weight=weight))
    if "keel_clearance" in design.quantities:
        terms.append(AtLeast("keel_clearance", 0.10, weight=weight, scale=0.02))
    return terms
