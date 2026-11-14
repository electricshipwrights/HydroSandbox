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

- Hull in the loop (`hydrosandbox/hull/`): first stage of porting boatmath
  (`hullgeom`, `thinship`) to differentiable Python.
  - `Hull`: offsets table (from numbers, mesh, or boatmath hull-record geometry),
    hydrostatics / ITTC-57 / thin-ship sources / transom closure at any
    sinkage and trim. Matches boatmath's Rust output on Wigley (Rw 0.6%, Rv 0.05%;
    reference numbers from `cargo run -p thinship --example wigley_curve`, built
    in a `rust:1-slim` container since there is no Rust toolchain on this box).
  - `CraftBuildup`: hull + foil assembly as one rigid craft; shared wave
    spectrum (interference), F_z / M_y residuals for sinkage and trim.

Left (hull port):
- Hull's own dynamic sinkage/trim/lift (boatmath `squat`, `nearfield`, 2D+t
  `lift`) not ported: attitude sees hydrostatics, foils and thrust line only.
- Slow: ~15-35 s per equilibrium solve (800 hull cells in the wave integral).
  Pre-integrating depth per station, as boatmath does, is the obvious speedup.
- No real hull tested (no CAD fixtures on this machine); only Wigley shapes.
- Transom closure ported but validated only for ordering, not against boatmath.
- Hull near field at the foil, and foil pressure on the hull, not modelled.

Left:
- Free-surface effect on *lift* is still the image approximation (kappa); a
  finite-Froude near field (cf. boatmath `thinship::nearfield`) is not ported.
- Buildup wave quadrature uses default length scales when geometry is symbolic;
  pass `wave_length_scales` for unusual sizes (large craft, very shallow foils).
- A foil lying exactly on the surface gives a large, meaningless wave drag in
  VLM/LL (linear theory invalid there; hard clipping).
- `junction_interference_drag` misquotes Hoerner (`/(tc+0.02)**3` should be `/tc**2`).
- Free surface is fixed at constant z in geometry axes (pitches with alpha; no heel).
- Junction drag is buildup-only, cavitation penalty defaults differ, VLM is
  inviscid; HydroLiftingLine is linear despite its docstring.
- Large copy-adapted overlap with AeroBuildup/VLM/LiftingLine; relies on private ASB methods.
