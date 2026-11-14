# NOTES

## hydrosandbox (branch `hydrosandbox`)

Gemini-authored hydrofoil package on top of AeroSandbox; reviewed 2026-11-14.

Done:
- Baseline import committed unmodified, then three review fixes:
  - `free_surface_induced_drag_factor` / `free_surface_lift_slope_factor` now use
    Prandtl's biplane factor with gap = 2h (matches HydroVLM within ~0.04).
  - HydroVLM `panel_Cpmin` is derived from strip-integrated section Cl
    (mesh-convergent) instead of leading-edge panel loading.
  - HydroVLM panel t/c no longer uses an `id()`-keyed cache (stale hits across foils).
- Regression tests added for all three; example STEP regenerated.

- Wave-making drag added to all three solvers (`library/wave_resistance.py`,
  `hydrodynamics/wave_drag.py`), ported from boatmath's `thinship` Michell
  conventions into differentiable Python (boatmath is Rust; not a dependency).
  - Sources for thickness (struts = Michell thin ship, foils, pods), dipoles for
    lift and side force, one spectrum so interference is included.
  - Drag due to lift = rigid-lid-image vortex drag + R_w. R_w -> 2x the phi=0
    image mutual drag as Fr -> inf, so it must not be added on top of the
    high-Froude image induced drag (the solvers swap the image term out).
  - Validated: Wigley closed form (0.13%), 2D vortex (<0.5%), infinite-Froude
    image limit (0.02% kernel, 0.4% through VLM/LL).

- Follow-up review fixes, merged from branch `hydrosandbox-fixes`:
  - `wing_thickness_wave_sources` now interpolates chord and thickness jumps
    onto the submerged `[t_lo, t_hi]` sub-interval (fixing tapered piercing struts).
  - `kelvin_wave_resistance` infers quadrature scales from any numeric
    coordinates in partially symbolic `Opti` graphs. (A symbolic `velocity`
    still requires `reference_velocity`: the branch's 5 m/s default was dropped
    on merge.)
  - `HydroBuildup._build_wave_lift_elements` projects elliptical weights onto
    vertical (`|u_z|`) vs lateral (`|u_y|`) lifting spans so multi-section
    winglet / C-foil tips do not mix vertical lift and tip sideforce.
  - Fixed `junction_interference_drag` exponent to Hoerner Ch. 8 Eq. 2
    (`/ tc**2`) and re-exported wave-resistance and biplane helpers at top level.

- Hull in the loop (`hydrosandbox/hull/`): port of boatmath (`hullgeom`,
  `thinship`) to differentiable Python.
  - `Hull`: offsets table (from numbers, mesh, or boatmath hull-record geometry);
    hydrostatics, ITTC-57, thin-ship sources, transom closure, and the dynamic
    sinkage force / trim moment (`thinship::squat`) at any sinkage and trim.
  - `CraftBuildup`: hull + foil assembly as one rigid craft; shared wave
    spectrum (interference); `solve_equilibrium` (Newton, 5-15 s) or Opti
    constraints (limited-memory Hessian; start from the Newton attitude).
  - Validated against boatmath's Rust output. Wigley: Rw 0.6%, Rv 0.05%,
    dynamic force 0.3%, moment 0.5%. Real hull (`~/hummingbird.igs`, cut by
    `boatmath hull`, free attitude, Fn 0.3-0.9): sinkage 0.2 mm, trim 0.06 deg,
    Rt 1.5%. References came from a `rust:1-slim` container (no Rust toolchain
    on this box); the built CLI and outputs were in the session scratchpad and
    are not kept. The IGES and its hull record are not committed.
  - Bugs found on the way, now tested: the transom hollow must stay level
    (not pitch with trim); transom depth uses the widest *immersed* breadth;
    CasADi reshape is column-major.

- Static pitch stability: `CraftBuildup.run_with_stability()` gives heave-free
  pitch stiffness and effective `GM_L`. `examples/optimize_foils_on_hull.py`
  minimizes drag over foil layout, CG shift, mass and thrust-line height with
  attitude equilibrium and a `GM_L` floor (multi-start; ~1.5 min per solve
  without the floor, several minutes with it).
- Context from Avi (2026-11-14): the target is power boats, so thrust-line
  position is a design freedom; CG and mass are design variables within a
  range. The hummingbird runs use placeholder loading (91 kg, CG over the
  centre of buoyancy, one demihull), not the real boat's.

- Merged `hydrosandbox-fixes` and `electric-motor` (propulsion: B-series,
  motors, hull-propeller interaction) into `hydrosandbox`. Fixed on merge:
  `hull_propeller_interaction` mixed degrees and radians for trim (degrees now).
- `PoweredCraftBuildup` (propulsion/powered_craft.py) couples the two: drive
  masses as point masses on the craft, thrust at the propeller along the shaft,
  propulsion sized at the craft's thrust and attitude. `solve()` iterates
  attitude -> RPM -> masses (~45 s on a 61x41 hull); residuals for Opti.
  `CraftBuildup` gained `point_masses` and `shaft_angle`.

- Catamarans: `CraftBuildup(demihull_spacing=...)`. Checked against boatmath on
  the NACRA cat (`~/nacra.igs`, 2.2 m spacing, 520 kg): Rt within 1.3%, sinkage
  0.3 mm, trim 0.05 deg, Fn 0.2-0.9, hull-hull interference included.
- `hsb.parallel_map` runs independent solves on all cores (the box has 24);
  `examples/optimize_foils_on_hull.py --jobs`.
- `local/` (git-excluded) holds the NACRA case: boatmath records and reference
  results, `nacra_check.py`, `nacra_baseline.py`, `nacra_cg_sweep.py`.

- Merged `audit-fixes` (true trimmed waterplane for foils, blunt-bow sources,
  thrust deduction, AD stability derivatives, numerical guards).
- Pitch stability as the design goal (Avi, 2026-11-14: not drag alone; stiffness,
  ideally with damping). `run_with_stability` adds `c_pitch_foils` (foils' pitch
  damping; the hull's own damping and added inertia are NOT modelled).
  `optimize()` in the example takes `objective` = drag / pitch_stiffness /
  pitch_damping, `drag_cap`, `damping_floor`. `Hull.dynamic_force_table` +
  `CraftBuildup(hull_dynamics=...)` make solves lighter (still ~5 GB, 5-16 min
  each with stability; keep to ~5 parallel workers on this 31 GB box).
- NACRA cat result (6 m/s, drag <= bare 392 N): stiffness and damping objectives
  give the same design (foil 1.8 m aft of CG, full span, CG at aft limit);
  GM_L 14.6 -> 23.6 m. Foil position, span and CG all sit on bounds I chose.
- `hsb.export_craft_step`: hull surfaces (designer's B-spline patches, or the
  offsets as bilinear patches) + foil solids in one STEP file.
- boatmath observation: with an immersed transom, `squat`'s trim moment does not
  shift with its reference by force x distance (the hollow adds sources but no
  weights). Ported faithfully; flagged to Avi.

- Hull-foil local interaction (`include_hull_foil_interaction`, default on):
  `Hull.induced_velocity` (hull flow at the foil) and `Hull.pressure_load`
  (foil's pressure on the hull). Small on the NACRA's slender hulls: <=0.2 deg
  of inflow change; a T-foil 0.35 m under a hull loses ~3% of its lift to hull
  suction (6% at 0.10 m clearance).
- Catamaran foil layouts in the example: `layout="pi"` (one foil, two struts)
  and `"twin"` (a T-foil under each hull); `Hull.keel_height`.
- Avi (2026-11-14): CG may go up to 1.0 m aft; wants two-strut and twin-T
  layouts tried, and hull-foil interaction modelled.
- Hull wave field at the foils (`include_hull_wave_field`, default on;
  `Hull.wave_field`, port of boatmath `thinship::nearfield`): elevation within
  a few mm of boatmath's field grids. It enters the foil as incidence only (the
  far-field integral already holds the mutual wave drag), and the elevation is
  taken at the main foil only. On the NACRA: up to ~1 deg inflow change and
  ~-70 mm elevation near the stern, cutting a fixed foil's lift 10-16%.
- Merged `memory-opt` (another session's work): `Hull.wave_amplitude_table` +
  `CraftBuildup(hull_wave_amplitude=...)`, hull state computed once per attitude
  (`_hull_state`, now also holding the stations and wave field the interaction
  needs), chunked wave integral, `optimize(use_hull_tables=True)`,
  `run_with_stability(include_pitch_damping=...)`, BLAS threads pinned in
  `parallel_map`. 93 tests pass. The `local/` study scripts do not use the
  tables yet; per-solve memory after the merge not yet measured.
- Layout study with the wave field (6 m/s, drag <= bare, CG -0.6..+1.0):
  `local/layouts_6ms.txt`.

Left (hull port):
- `examples/optimize_foils_on_hull.py` has been run only for the no-floor,
  fixed-loading case on the hummingbird hull; the full study (GM_L floors, CG and
  thrust-height variables) has not been run. It does not use the propulsion model.
- Stability is static only: no pitch inertia, added mass or damping, so no
  porpoising or motion-in-waves prediction.
- boatmath's 2D+t planing `lift` not ported.
- The foil's waves do not enter the hull's dynamic force; the foil's field is not
  reflected in the hull.
- No hull pitch damping or added inertia (needs boatmath's seakeeping).
- Default 61x41 offsets give ~2-5% on a shallow real hull; finer grids converge
  to boatmath (121x161: <1% Rw, <2% force).
- IPOPT needs a good start for the attitude variables; a cold start can report
  infeasible.

Left:
- Free-surface effect on *lift* is still the image approximation (kappa); a
  finite-Froude near field (cf. boatmath `thinship::nearfield`) is not ported.
- A foil lying exactly on the surface gives a large, meaningless wave drag in
  VLM/LL (linear theory invalid there; hard clipping).
- Free surface is fixed at constant z in geometry axes (pitches with alpha; no heel).
- Junction drag is buildup-only, cavitation penalty defaults differ, VLM is
  inviscid; HydroLiftingLine is linear despite its docstring.
- Large copy-adapted overlap with AeroBuildup/VLM/LiftingLine; relies on private ASB methods.
