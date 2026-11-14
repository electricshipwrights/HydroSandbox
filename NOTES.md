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

Left:
- Wave-making drag is missing entirely. Plan: Michell thin-ship theory, building on
  the `boatmath` repo (not yet checked out on this machine).
- `junction_interference_drag` misquotes Hoerner (`/(tc+0.02)**3` should be `/tc**2`).
- Free surface is fixed at constant z in geometry axes (pitches with alpha; no heel).
- Solvers disagree on drag contents (junction drag buildup-only, cavitation penalty
  defaults differ, VLM inviscid); HydroLiftingLine is linear despite its docstring.
- VLM/LL clip hard at the surface (discontinuous as a foil broaches).
- Large copy-adapted overlap with AeroBuildup/VLM/LiftingLine; relies on private ASB methods.
