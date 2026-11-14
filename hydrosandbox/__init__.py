"""HydroSandbox: a Python package for design optimization of lifting hydrofoil assemblies."""

from pathlib import Path
import aerosandbox.numpy as numpy
from aerosandbox import (
    Airfoil,
    ControlSurface,
    Fuselage,
    FuselageXSec,
    KulfanAirfoil,
    MassProperties,
    Opti,
    OptiSol,
    Wing,
    WingXSec,
)
from hydrosandbox import fluid, geometry, hull, hydrodynamics, library, performance
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import (
    Foil,
    HydroFuselage,
    HydrofoilAssembly,
    Strut,
    clip_wing_to_free_surface,
    export_assembly_step,
    fuselage_submerged_volume_and_centroid,
    wing_submerged_volume_and_centroid,
)
from hydrosandbox.hull import CraftBuildup, Hull
from hydrosandbox.hydrodynamics import (
    HydroBuildup,
    HydroLiftingLine,
    HydroVLM,
    calculate_induced_velocity_horseshoe_hydro,
    cavitation_bucket,
    evaluate_hydrofoil_2D,
)
from hydrosandbox.library import (
    cavitation_drag_increment,
    cavitation_inception_speed,
    cavitation_margin,
    cavitation_number,
    estimate_Cpmin_inviscid,
    free_surface_induced_drag_factor,
    free_surface_lift_slope_factor,
    froude_image_weight,
    junction_interference_drag,
    smooth_submergence_fraction,
    smooth_surface_piercing_weight,
    spray_drag_strut,
    ventilation_margin_lifting_foil,
    ventilation_margin_strut,
)
from hydrosandbox.performance import HydroOperatingPoint

_hsb_root = Path(__file__).parent

try:
    from importlib.metadata import version

    __version__ = version("AeroSandbox")
except Exception:
    __version__ = "unknown"


def run_tests() -> int:
    """Run the HydroSandbox unit test suite using pytest."""
    try:
        import pytest
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Please install `pytest` (`pip install pytest`) to run HydroSandbox unit tests."
        )

    import matplotlib.pyplot as plt

    with plt.ion():
        return pytest.main([str(_hsb_root / "tests")])


__all__ = [
    "fluid",
    "geometry",
    "hull",
    "hydrodynamics",
    "library",
    "performance",
    "numpy",
    "Water",
    "HydroOperatingPoint",
    "Foil",
    "Strut",
    "HydroFuselage",
    "HydrofoilAssembly",
    "Hull",
    "CraftBuildup",
    "clip_wing_to_free_surface",
    "export_assembly_step",
    "wing_submerged_volume_and_centroid",
    "fuselage_submerged_volume_and_centroid",
    "HydroVLM",
    "HydroLiftingLine",
    "HydroBuildup",
    "evaluate_hydrofoil_2D",
    "cavitation_bucket",
    "calculate_induced_velocity_horseshoe_hydro",
    "smooth_submergence_fraction",
    "smooth_surface_piercing_weight",
    "froude_image_weight",
    "free_surface_lift_slope_factor",
    "free_surface_induced_drag_factor",
    "spray_drag_strut",
    "junction_interference_drag",
    "cavitation_number",
    "cavitation_margin",
    "cavitation_inception_speed",
    "estimate_Cpmin_inviscid",
    "cavitation_drag_increment",
    "ventilation_margin_lifting_foil",
    "ventilation_margin_strut",
    "Opti",
    "OptiSol",
    "Airfoil",
    "KulfanAirfoil",
    "Wing",
    "WingXSec",
    "Fuselage",
    "FuselageXSec",
    "ControlSurface",
    "MassProperties",
    "run_tests",
    "__version__",
]
