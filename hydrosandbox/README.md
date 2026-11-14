# HydroSandbox (`hydrosandbox`)

**HydroSandbox** adapts AeroSandbox's CasADi/NumPy algorithmic differentiation (`aerosandbox.numpy`), `Opti` optimization interface, meshing utilities, and NeuralFoil 2D section aerodynamics to **lifting hydrofoil assemblies** (main foils, stabilizers, surface-piercing masts/struts, and submerged pods/fuselages).

## Package Structure

- `hydrosandbox.fluid`: [`Water`](fluid/water.py) thermodynamic & physical property model (UNESCO EOS-80 seawater/freshwater density, Sharqawy/ITTC 2011 viscosity, vapor pressure, hydrostatic depth pressure, speed of sound, surface tension).
- `hydrosandbox.performance`: [`HydroOperatingPoint`](performance/operating_point.py) operating condition with free-surface elevation `surface_z`, Bernoulli `total_pressure`, depth/chord Froude numbers, Weber number, and local cavitation number `sigma`.
- `hydrosandbox.library`: Smooth free-surface physics ([`free_surface.py`](library/free_surface.py): smooth submergence, method-of-images Froude weights, free-surface lift/drag factors, Hoerner spray drag, T-junction interference drag) and cavitation/ventilation models ([`cavitation.py`](library/cavitation.py)).
- `hydrosandbox.geometry`: [`Foil`](geometry/components.py), [`Strut`](geometry/components.py), [`HydroFuselage`](geometry/components.py), [`HydrofoilAssembly`](geometry/assembly.py), free-surface clipping (`clip_wing_to_free_surface`), and Archimedes hydrostatics (`displaced_volume`, `center_of_buoyancy`, `hydrostatic_forces`).
- `hydrosandbox.hydrodynamics`:
  - [`HydroBuildup`](hydrodynamics/hydro_buildup.py): Fast, smooth, CasADi-differentiable 3D conceptual hydrodynamic buildup for `Opti` design optimization.
  - [`HydroVLM`](hydrodynamics/vortex_lattice_method.py): 3D Vortex Lattice Method with free-surface clipping and method-of-images boundary conditions (`"high_froude"`, `"froude_blended"`, `"rigid_lid"`, `"none"`).
  - [`HydroLiftingLine`](hydrodynamics/lifting_line.py): Nonlinear viscous 3D lifting-line solver coupling horseshoe downwash with incompressible NeuralFoil section polars (`mach=0.0`) and `Cpmin_0` cavitation diagnostics.
  - [`evaluate_hydrofoil_2D`](hydrodynamics/hydro_2D.py) & [`cavitation_bucket`](hydrodynamics/hydro_2D.py): 2D section polar evaluation and cavitation-free speed buckets.

## Wave-Making Drag

All three 3D solvers include steady wave-making drag by default (`include_wave_drag=True`, active whenever `op_point.surface_z` is set and `free_surface_mode != "none"`), using linear far-field theory in [`library/wave_resistance.py`](library/wave_resistance.py):

- **Thickness** of struts, foils and pods radiates as sources. For a surface-piercing strut this is Michell's thin-ship integral.
- **Lift and side force** radiate as vertical and horizontal pressure dipoles at the depth of the loading.
- Everything is superposed in one free-wave spectrum, so interference (foil with stabilizer, strut with foil) is included.

The formulation and conventions follow the `thinship` crate of the `boatmath` project, and the kernel is validated against the same closed forms (the Wigley hull's Michell integral, Havelock's 2D results).

Results report `D_induced` as the trailing-vortex drag with a **rigid-lid** image and `D_wave` as the wave resistance. Their sum is the drag due to lift; as `Fr -> infinity` it recovers the classical `phi = 0` image-method value, so `D_wave` is not an extra term on top of a high-Froude induced drag. The chosen `free_surface_mode` still sets how the free surface affects *lift*.

Limits: deep water, linear theory (no breaking or spray sheets, loading not closer than about half a chord to the surface), and the free surface's effect on lift is still the image approximation rather than the full finite-Froude near field.

## Hull in the Loop

[`hydrosandbox.hull`](hull/) ports the core of the `boatmath` project's Rust crates (`hullgeom`, `thinship`) to `aerosandbox.numpy`, so a hull can sit inside the same `Opti` problem as the foils:

- [`Hull`](hull/hull.py): a hull as a table of offsets (from numbers, a triangle mesh, or a boatmath `hull` record's mesh or B-spline geometry). At any sinkage and trim it gives hydrostatics, ITTC-57 friction, and thin-ship wave sources with boatmath's transom closure. Checked against boatmath itself on the Wigley hull: wave resistance within 0.6%, friction within 0.05%.
- [`CraftBuildup`](hull/craft.py): hull plus `HydrofoilAssembly` as one rigid craft. The hull's sources and the foils' lift dipoles share one wave spectrum, so hull-foil wave interference is in `D_wave` (and reported as `D_wave_interference`). It returns the net vertical force and pitching moment about the centre of gravity; make sinkage and trim `Opti` variables and constrain both to zero, or call `CraftBuildup.solve_equilibrium(...)` for a fixed design.

```python
hull = hsb.Hull.wigley(length=5.0, beam=0.8, draft=0.25, freeboard=0.3)
res = hsb.CraftBuildup.solve_equilibrium(
    hull=hull, assembly=assembly, velocity=4.5, mass=456.0,
    xyz_cg=[0.2, 0.0, 0.1], xyz_thrust=[2.0, 0.0, -0.3],
)
res["trim"], res["sinkage"], res["D"], res["D_wave_interference"]
```

Not ported yet: the hull's own dynamic sinkage, trim and lift (boatmath's `squat`, `nearfield` and 2D+t `lift`), so the attitude responds to hydrostatics, the foils and the thrust line only. The hull's near-field flow at the foil and the foil's pressure on the hull are also not modelled.

## Coordinate & Sign Conventions

- **Geometry axes (`_g`)**: `+x_g` aft/downstream, `+y_g` starboard/right, `+z_g` **upward** (so buoyancy acts in `+z_g`).
- **Free surface (`surface_z`)**: `op_point.surface_z` specifies the waterplane elevation in geometry `z_g` coordinates [m]. Local submergence depth is `h = surface_z - z_g` (`h > 0` underwater, `h < 0` emerged in air). Setting `surface_z=None` models an unbounded fluid using `water.depth - z_g`.

## Quick Start

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

## Electric Propulsion

`hydrosandbox.propulsion` pairs smooth, CasADi-differentiable Wageningen B-series propeller models (Keller & Burrill cavitation, Reynolds & Gutsche oblique-flow corrections), Drela 3-parameter and 4-term polynomial loss electric motor models, a vendored 159-motor catalog (plus small waterproof foil-assist pod motors), thin-ship Rankine wake + Lagally thrust deduction (`hull_propeller_interaction`), and coupled `ElectricPropulsionBuildup` analyses:

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

## Example Optimization & Tests

Run the coupled T-foil MDO example:

```bash
python hydrosandbox/examples/optimize_hydrofoil_assembly.py
```

Run the unit test suite:

```bash
pytest hydrosandbox/tests -v
```
