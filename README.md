# HydroSandbox

> **Built on [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)**: HydroSandbox is a standalone package for differentiable hydrofoil, hull, and electric marine propulsion design optimization that builds directly on Peter Sharpe's [**AeroSandbox**](https://github.com/peterdsharpe/AeroSandbox) ([documentation](https://peterdsharpe.github.io/AeroSandbox/)). Rather than vendoring a fork, HydroSandbox installs `aerosandbox` (`>=4.2.10`) as a dependency and uses its CasADi/NumPy automatic-differentiation backend (`aerosandbox.numpy`), `Opti` IPOPT nonlinear optimization interface, parametric 3D geometry primitives (`Wing`, `Fuselage`, `Airfoil`, `KulfanAirfoil`, `MassProperties`), and [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) 2D viscous airfoil surrogates—extending them with marine physics ported from `boatmath` (`hullgeom`, `thinship`) and empirical marine propulsion models.

HydroSandbox targets gradient-based **multidisciplinary design optimization (MDO)** of foil-assisted boats and electric marine craft in seconds on a laptop. All solvers combine smooth, CasADi-differentiable potential flow, linearized thin-ship / Kelvin–Havelock wave resistance, and semi-empirical hydrodynamic/propulsion buildups rather than mesh-based RANS/VOF Navier–Stokes CFD.

---

## Capabilities

### 1. Foil, Strut & Pod Hydrodynamics
- **Fluid & Operating Environment** (`hydrosandbox.fluid`, `hydrosandbox.performance`):
  - `Water`: UNESCO EOS-80 seawater/freshwater density, Sharqawy/ITTC 2011 dynamic viscosity, vapor pressure, hydrostatic depth pressure, speed of sound, and surface tension.
  - `HydroOperatingPoint`: operating state with waterplane elevation `surface_z`, depth/chord Froude numbers ($Fn_h$, $Fn_c$), Weber number, Bernoulli total pressure, and local cavitation number $\sigma$.
- **Free-Surface, Cavitation & Junction Physics** (`hydrosandbox.library`, `hydrosandbox.hydrodynamics`):
  - Method-of-images free-surface boundary conditions (`free_surface_mode="high_froude" | "froude_blended" | "rigid_lid" | "none"`), with `froude_image_weight` smoothly transitioning across $Fn_h \approx 1$.
  - Smooth free-surface submergence clipping and Archimedes hydrostatics (`clip_wing_to_free_surface`, `wing_submerged_volume_and_centroid`, `fuselage_submerged_volume_and_centroid`).
  - NeuralFoil incompressible 2D section polars (`evaluate_hydrofoil_2D`, `cavitation_bucket`) coupled with inviscid $C_{p,\min}$, cavitation inception speed/margin, smooth cavitation drag increment, foil/strut ventilation margins, Hoerner strut spray drag, and T-junction interference drag.
  - Three 3D hydrodynamic solvers: `HydroBuildup` (fast conceptual buildup built for `Opti`), `HydroVLM` (3D vortex lattice with image reflections), and `HydroLiftingLine` (nonlinear viscous 3D lifting line).

### 2. Linearized Wave-Making & Near-Field Wave Flow
- **Unified Far-Field Wave Spectrum** (`hydrosandbox.library.wave_resistance`):
  - `kelvin_wave_resistance`, `wing_thickness_wave_sources`, `fuselage_wave_sources`, `element_source_amplitude`, `merge_wave_elements`, and `wave_quadrature_nodes` compute steady deep-water Kelvin/Havelock wave resistance in one combined spectrum across hulls, lifting/dihedral foils (lift dipoles + thickness sources), surface-piercing struts (Michell thin-ship sources), and submerged pods.
  - Reports rigid-lid vortex drag (`D_induced`) alongside `D_wave` so their sum approaches the high-Froude image limit without double-counting wave drag in induced drag.
- **Wave Elevation & Orbital Velocity** (`wave_field_from_source_amplitude` in `hydrosandbox.hull.hull`):
  - Evaluates symmetric free-surface wave elevation $\zeta(x, y)$ and orbital velocity perturbations $(u, v, w)$ induced by any combined set of hull, foil, strut, and pod wave singularities.

### 3. Hull in the Loop & Coupled Craft Balance
- **Thin-Ship Hull Model** (`hydrosandbox.hull.hull`):
  - Constructible from offset tables (`Hull(...)`), analytical Wigley hulls (`Hull.wigley`), triangle meshes (`Hull.from_mesh`), or `boatmath` geometry records (`Hull.from_boatmath_geometry`), supporting both monohulls and symmetric catamarans (`demihull_spacing`).
  - Evaluates panel Archimedes hydrostatics, ITTC-57 wetted-area skin friction, Froude-tapered dry-transom hollow closure, and linearized thin-ship dynamic sinkage force & trim moment (`Hull.dynamic_force`).
- **Two-Way Hull–Foil Coupling & Attitude Tables** (`Hull`, `CraftBuildup`):
  - Models thin-ship hull thickness velocity at foil control points (`Hull.induced_velocity`), image-corrected foil horseshoe suction on hull bottom panels (`Hull.pressure_load`), and the hull's free-surface wave elevation and orbital velocity at downstream foils (`Hull.wave_field`).
  - `Hull.dynamic_force_table()` (`DynamicForceTable`) and `Hull.wave_amplitude_table()` (`WaveAmplitudeTable`) precompute cubic CasADi B-spline surrogates over `(sinkage, trim)` at a target speed so IPOPT evaluates hull forces and wave spectra in microseconds without inflating symbolic expression graphs.
- **Equilibrium & Longitudinal Stability** (`hydrosandbox.hull.craft`):
  - `CraftBuildup` solves rigid-body vertical-force (`F_z_net == 0`) and pitching-moment (`M_y_net == 0`) equilibrium about `xyz_cg` (via `CraftBuildup.solve_equilibrium` for fixed designs or as constraints inside `asb.Opti`), accepts arbitrary `inflow_perturbation` callbacks, and provides `run_with_stability()` returning heave-free longitudinal metacentric height `GM_L = k_pitch / (m * g)` and quasi-steady foil pitch damping `c_pitch_foils`.

### 4. Electric Marine Propulsion & Propeller–Foil Interaction
- **Propeller & Motor Models** (`hydrosandbox.propulsion`):
  - Smooth, differentiable Wageningen B-series polynomials (`evaluate_bseries`, `optimize_bseries_propeller`, `sweep_bseries_rpm`) with Keller & Burrill cavitation margins and Lerbs Reynolds corrections.
  - Continuous Drela (`motor_performance_drela`) and 4-term polynomial loss (`motor_performance_loss`) electric motor models, drivetrain/battery mass regressions (`mass_motor_marine`, `mass_controller_marine`, `mass_gearbox_marine`, `mass_battery_marine`), and a vendored 159-motor catalog plus waterproof foil-assist pod motors (`MotorDatabase.vendored(include_foil_assist_pods=True)`).
- **Coupled Wake, Suction & Propeller–Foil Flow**:
  - `hull_propeller_interaction` computes Rankine-ovoid + boundary-layer wake fraction and Lagally point-sink thrust deduction.
  - `hydrosandbox.propulsion.interaction` provides `actuator_disc_velocity` (sink-disc inflow + vortex-cylinder jet and swirl) and `foils_at_propeller` (foil/strut viscous wake momentum deficit + horseshoe-vortex induced velocity at the disc).
  - `ElectricPropulsionBuildup` and closed-loop `PoweredCraftBuildup` couple drivetrain weight, CG shift, thrust line height/shaft inclination (`xyz_propeller`, `shaft_angle`), wake/thrust deduction, and endurance battery sizing back into craft equilibrium.

### 5. CAD & ParaView Flow-Field Export
- **STEP Export** (`hydrosandbox.geometry.step_export`): Zero-dependency native AP214 STEP B-Rep export for foil assemblies and craft (`export_assembly_step`, `export_craft_step`, or optional `cadquery` backend).
- **ParaView VTK Export** (`hydrosandbox.geometry.vtk_export`): `export_craft_vtk` writes a multi-block `.vtm` dataset alongside `<prefix>_foils.vtp` (surface pressure and NeuralFoil boundary-layer diagnostics), `<prefix>_hull.vtp` (hull surface pressure), `<prefix>_surface.vtp` (combined hull + foil + pod wave elevation), and `<prefix>_bodies.vtp`.

---

## Installation

HydroSandbox requires Python `>=3.11` and automatically installs `aerosandbox>=4.2.10`:

```bash
git clone https://github.com/electricshipwrights/HydroSandbox.git
cd HydroSandbox
uv sync --extra full --extra test
```

Or using `pip`:

```bash
pip install -e ".[full,test]"
```

Run only core runtime dependencies with `pip install -e .`, and run tests with:

```bash
pytest hydrosandbox/tests -v
```

---

## Quick Start

### 1. Hydrofoil Assembly Analysis

```python
import aerosandbox as asb
import hydrosandbox as hsb

water = hsb.Water.seawater(temperature_C=18.0)
op = hsb.HydroOperatingPoint(water=water, velocity=12.0, alpha=2.0, surface_z=0.0)

foil = hsb.Foil(
    name="Main Foil",
    symmetric=True,
    xsecs=[
        asb.WingXSec(xyz_le=[0.0, 0.0, -0.6], chord=0.14, airfoil=asb.Airfoil("naca2412")),
        asb.WingXSec(xyz_le=[0.01, 0.45, -0.6], chord=0.09, airfoil=asb.Airfoil("naca2412")),
    ],
)
strut = hsb.Strut.from_dimensions(
    root_xyz=[0.02, 0.0, -0.6], length=0.9, root_chord=0.14, tip_chord=0.11
)
assembly = hsb.HydrofoilAssembly(name="T-Foil", foils=[foil], struts=[strut])

buildup_res = hsb.HydroBuildup(assembly=assembly, op_point=op).run_with_stability_derivatives()
vlm_res = hsb.HydroVLM(assembly=assembly, op_point=op).run()
ll_res = hsb.HydroLiftingLine(assembly=assembly, op_point=op).run()
bucket = hsb.cavitation_bucket(asb.Airfoil("naca2412"), velocity=12.0, chord=0.14, depth=0.6, water=water)
```

### 2. Hull + Foils Equilibrium & Stability

```python
import hydrosandbox as hsb

hull = hsb.Hull.wigley(length=10.0, beam=1.0, draft=0.625)
res = hsb.CraftBuildup.solve_equilibrium(
    hull=hull,
    velocity=4.5,
    mass=1800.0,
    xyz_cg=[0.0, 0.0, 0.0],
)
print(res["trim"], res["sinkage"], res["D"], res["D_wave"])
```

### 3. Coupled Electric Propulsion

```python
import hydrosandbox as hsb

op = hsb.HydroOperatingPoint(velocity=6.0)
hull = hsb.Hull.wigley(length=10.0, beam=1.0, draft=0.625)
db = hsb.MotorDatabase.vendored(include_foil_assist_pods=True)
motor = db.motor("flipsky_6384_120kv_36v")

buildup = hsb.ElectricPropulsionBuildup(
    op_point=op,
    drag_bare_N=80.0,
    hull=hull,
    disc_xyz=[5.2, 0.0, -0.35],
    propeller_diameter=0.18,
    propeller_pitch_ratio=1.2,
    propeller_ear=0.45,
    propeller_rpm=1500.0,
    gearbox_ratio=2.5,
    motor=motor,
    endurance_hours=1.5,
)
res = buildup.run()
```

For full closed-loop vehicle sizing (coupled mass/CG/trim/thrust/battery iteration), use `hsb.PoweredCraftBuildup.solve(...)`.

---

## Optimization Examples

- **Standalone Stern T-Foil MDO** ([`hydrosandbox/examples/optimize_hydrofoil_assembly.py`](hydrosandbox/examples/optimize_hydrofoil_assembly.py)): Optimizes span, root/tip chords, and incidence with cavitation/ventilation safety margins, cross-checks against `HydroVLM` and `HydroLiftingLine`, and optionally exports STEP geometry:
  ```bash
  python hydrosandbox/examples/optimize_hydrofoil_assembly.py --velocity 6.0 --mass 30.0
  ```
- **Hull-in-the-Loop Foil & Pod Optimization** ([`hydrosandbox/examples/optimize_foils_on_hull.py`](hydrosandbox/examples/optimize_foils_on_hull.py)): Minimizes total craft drag (hull friction + wave interference + foil/strut/pod drag) across monohull (`layout="centreline"`) and catamaran (`layout="pi" | "twin"`) configurations subject to attitude equilibrium, longitudinal stability floors (`GM_L` and `c_pitch_foils`), optional section thickness/camber optimization (`--optimize-section`) with solid-section root bending-stress limits (`--allowable-stress`), optional steerable/fixed pod drives (`EPROPULSION_POD_3`), STEP/VTK export, and multi-start search via `hsb.parallel_map`:
  ```bash
  python hydrosandbox/examples/optimize_foils_on_hull.py --velocity 4.5 --gml-fractions 0.6,0.8,1.0 --stabilizer
  ```

---

## Package Layout

```text
hydrosandbox/
├── fluid/          # Seawater/freshwater thermodynamic & transport properties (Water)
├── geometry/       # Foil, Strut, HydroFuselage, HydrofoilAssembly, clipping, STEP & VTK export
├── hull/           # Hull (thin-ship, hydrostatics, wave field, B-spline tables) & CraftBuildup
├── hydrodynamics/  # HydroBuildup, HydroVLM, HydroLiftingLine, 2D polars & cavitation buckets
├── library/        # Free-surface image factors, cavitation/ventilation, Kelvin wave resistance
├── performance/    # HydroOperatingPoint (depth Froude, Weber, cavitation numbers)
├── propulsion/     # B-series propellers, electric motor models/catalog, wake/foil interaction, PoweredCraftBuildup
├── examples/       # MDO scripts for standalone T-foils and hull-coupled hydrofoil craft
├── parallel.py     # Multi-process parallel_map worker helper (OpenBLAS-pinned per process)
└── tests/          # Unit and analytical/boatmath parity test suite
```

---

## Validation Summary

Kernel and hull formulations are tested against analytical closed forms (Havelock 2D dipoles, Wigley hull Michell integral) and the Rust `boatmath` (`hullgeom`, `thinship`, `propeller`) reference outputs:

| Case | Quantity | Agreement |
|---|---|---|
| Wigley hull, $Fn \in [0.2, 0.5]$ | Wave resistance, skin friction | 0.6%, 0.05% |
| Wigley hull, $Fn \in [0.2, 0.5]$ | Dynamic sinkage force, trim moment | 0.3%, 0.5% |
| Catamaran demihull (IGES, transom), free to sink & trim, $Fn \in [0.3, 0.9]$ | Sinkage, trim, total resistance | 0.2 mm, 0.06 deg, 1.5% |

---

## Coordinate Conventions & Modeling Limits

- **Geometry axes**: `+x` aft/downstream, `+y` starboard, `+z` upward (`surface_z = 0` is the calm free surface; submergence depth is `h = surface_z - z`, positive underwater).
- **Modeling assumptions**: Deep water, linearized free-surface wave theory (no wave breaking, spray sheets, or planing 2D+t hull lift), attached or mildly separated 2D sections represented by NeuralFoil, steady wake/actuator-disc propeller–foil interaction, and inception-margin cavitation constraints rather than multiphase sheet or supercavitation simulation. Foil wave reflection in the hull and foil waves inside the hull's own dynamic sinkage pressure field are not modeled.
