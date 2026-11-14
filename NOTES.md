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

- Follow-up review fixes on branch `hydrosandbox-fixes`:
  - `wing_thickness_wave_sources` now interpolates chord and thickness jumps
    onto the submerged `[t_lo, t_hi]` sub-interval (fixing tapered piercing struts).
  - `kelvin_wave_resistance` infers quadrature scales from any numeric
    coordinates in partially symbolic `Opti` graphs and defaults
    `reference_velocity` to `5.0` m/s when `velocity` is symbolic.
  - `HydroBuildup._build_wave_lift_elements` projects elliptical weights onto
    vertical (`|u_z|`) vs lateral (`|u_y|`) lifting spans so multi-section
    winglet / C-foil tips do not mix vertical lift and tip sideforce.
  - Fixed `junction_interference_drag` exponent to Hoerner Ch. 8 Eq. 2
    (`/ tc**2`) and re-exported wave-resistance and biplane helpers at top level.

Left:
- Free-surface effect on *lift* is still the image approximation (kappa); a
  finite-Froude near field (cf. boatmath `thinship::nearfield`) is not ported.
- A foil lying exactly on the surface gives a large, meaningless wave drag in
  VLM/LL (linear theory invalid there; hard clipping).
- Free surface is fixed at constant z in geometry axes (pitches with alpha; no heel).
- Junction drag is buildup-only, cavitation penalty defaults differ, VLM is
  inviscid; HydroLiftingLine is linear despite its docstring.
- Large copy-adapted overlap with AeroBuildup/VLM/LiftingLine; relies on private ASB methods.
