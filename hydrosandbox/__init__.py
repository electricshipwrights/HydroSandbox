"""HydroSandbox: differentiable hydrofoil, hull, and electric marine propulsion design optimization built on AeroSandbox."""

import os as _os
from pathlib import Path

# CasADi/IPOPT's bundled OpenBLAS otherwise reserves ~80 MB per CPU core on first
# solve (~2 GB on a 24-core machine), even though each Opti solve is single-threaded.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

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
from hydrosandbox import fluid, geometry, hull, hydrodynamics, library, performance, propulsion
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import (
    Foil,
    HydroFuselage,
    HydrofoilAssembly,
    Strut,
    clip_wing_to_free_surface,
    export_assembly_step,
    export_craft_step,
    export_craft_vtk,
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
    horseshoe_lattice_wave_drag,
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
    fuselage_wave_sources,
    junction_interference_drag,
    kelvin_wave_resistance,
    merge_wave_elements,
    prandtl_biplane_interference_factor,
    smooth_submergence_fraction,
    smooth_surface_piercing_weight,
    spray_drag_strut,
    ventilation_margin_lifting_foil,
    ventilation_margin_strut,
    wave_quadrature_nodes,
    wing_thickness_wave_sources,
)
from hydrosandbox.parallel import parallel_map
from hydrosandbox.performance import HydroOperatingPoint
from hydrosandbox.propulsion import (
    BSeriesInputs,
    CatalogMotor,
    ElectricPropulsionBuildup,
    PoweredCraftBuildup,
    MotorDatabase,
    TopOperatingPoint,
    evaluate_bseries,
    hull_propeller_interaction,
    mass_battery_marine,
    mass_controller_marine,
    mass_gearbox_marine,
    mass_motor_marine,
    match_motors_to_curve,
    motor_performance_drela,
    motor_performance_loss,
    optimize_bseries_propeller,
    propeller_actuator_disk_power,
    sweep_bseries_rpm,
)

_hsb_root = Path(__file__).parent

try:
    from importlib.metadata import version

    __version__ = version("hydrosandbox")
except Exception:
    __version__ = "0.1.0"


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
    "propulsion",
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
    "export_craft_step",
    "export_craft_vtk",
    "wing_submerged_volume_and_centroid",
    "fuselage_submerged_volume_and_centroid",
    "HydroVLM",
    "HydroLiftingLine",
    "HydroBuildup",
    "evaluate_hydrofoil_2D",
    "cavitation_bucket",
    "horseshoe_lattice_wave_drag",
    "calculate_induced_velocity_horseshoe_hydro",
    "smooth_submergence_fraction",
    "smooth_surface_piercing_weight",
    "froude_image_weight",
    "prandtl_biplane_interference_factor",
    "free_surface_lift_slope_factor",
    "free_surface_induced_drag_factor",
    "spray_drag_strut",
    "junction_interference_drag",
    "kelvin_wave_resistance",
    "wave_quadrature_nodes",
    "wing_thickness_wave_sources",
    "fuselage_wave_sources",
    "merge_wave_elements",
    "cavitation_number",
    "cavitation_margin",
    "cavitation_inception_speed",
    "estimate_Cpmin_inviscid",
    "cavitation_drag_increment",
    "ventilation_margin_lifting_foil",
    "ventilation_margin_strut",
    "evaluate_bseries",
    "propeller_actuator_disk_power",
    "BSeriesInputs",
    "TopOperatingPoint",
    "sweep_bseries_rpm",
    "optimize_bseries_propeller",
    "motor_performance_drela",
    "motor_performance_loss",
    "mass_motor_marine",
    "mass_controller_marine",
    "mass_gearbox_marine",
    "mass_battery_marine",
    "CatalogMotor",
    "MotorDatabase",
    "match_motors_to_curve",
    "hull_propeller_interaction",
    "ElectricPropulsionBuildup",
    "PoweredCraftBuildup",
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
    "parallel_map",
    "run_tests",
    "__version__",
]
