"""
Multidisciplinary design optimization (MDO) of a stern lift-assist inverted-T boat hydrofoil.

Optimizes main-foil planform (span, root chord, tip chord) and wing mounting incidence
(``wing_incidence``, leaving the vertical mast and junction fairing aligned with flow at
``alpha = 0 deg``) to minimize total hydrodynamic drag at cruise speed using AeroSandbox's
``Opti`` (CasADi + IPOPT) coupled to ``HydroBuildup``.

By default (``include_stabilizer=False``), models a single-wing stern lift-assist T-foil
assembly mounted to a boat hull:
- **Main Foil**: single continuous lifting wing at the base of the strut
- **Piercing Mast**: single vertical surface-piercing strut
- **Junction Fairing**: compact 6-station teardrop bulb fairing around the wing-mast root

Because the boat hull reacts pitching moments, standalone pitch trim and static stability
constraints are omitted unless ``include_stabilizer=True`` (``--include-stabilizer``) is passed.

Default target: ``velocity = 6.0 m/s``, ``craft_mass_kg = 30.0 kg`` (~294 N total
vertical lift + Archimedes buoyancy), ``submergence_depth = 0.35 m``. Pass ``--velocity``,
``--mass``, ``--depth``, ``--max-ar``, ``--min-taper``, or ``--export-step`` to customize.

After convergence, the optimal design is cross-checked with ``HydroVLM`` (3D free-surface
method-of-images VLM) and ``HydroLiftingLine`` (nonlinear viscous 3D lifting line).
"""

import argparse
import aerosandbox as asb
import aerosandbox.numpy as np
import hydrosandbox as hsb


def build_hydrofoil_assembly(
    half_span_main: float,
    chord_root_main: float,
    chord_tip_main: float,
    wing_incidence: float,
    xyz_ref: list[float],
    submergence_depth: float = 0.35,
    scale: float = 1.0,
    include_stabilizer: bool = False,
    stab_incidence: float = -2.0,
) -> hsb.HydrofoilAssembly:
    """
    Build a parameterized inverted-T boat hydrofoil assembly.

    Args:
        half_span_main: Main foil semi-span [m].
        chord_root_main: Main foil root chord [m].
        chord_tip_main: Main foil tip chord [m].
        wing_incidence: Main foil mounting incidence twist [deg].
        xyz_ref: Moment reference point ``[x, y, z]`` [m].
        submergence_depth: Main-foil submergence depth ``h`` below ``surface_z = 0`` [m].
        scale: Length scale factor ``(craft_mass_kg / 30.0) ** (1/3)`` used to size
            the strut chord/extension and junction fairing pod consistently.
        include_stabilizer: If ``False`` (default), builds a single-wing stern T-foil
            with a compact 6-station teardrop junction fairing. If ``True``, adds an aft
            ``Horizontal Stabilizer`` and extended boom pod for standalone trimmed flight.
        stab_incidence: Horizontal stabilizer incidence twist [deg] when
            ``include_stabilizer=True``.
    """
    z_foil = -submergence_depth
    z_pod = z_foil - 0.002 * scale
    main_airfoil = asb.Airfoil("naca0012")
    strut_airfoil = asb.Airfoil("naca0010")

    main_foil = hsb.Foil(
        name="Main Foil",
        symmetric=True,
        role="main",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.0, 0.0, z_foil],
                chord=chord_root_main,
                twist=wing_incidence,
                airfoil=main_airfoil,
            ),
            asb.WingXSec(
                xyz_le=[
                    0.25 * (chord_root_main - chord_tip_main),
                    half_span_main,
                    z_foil,
                ],
                chord=chord_tip_main,
                twist=wing_incidence,
                airfoil=main_airfoil,
            ),
        ],
    )

    strut = hsb.Strut.from_dimensions(
        name="Piercing Mast",
        root_xyz=[0.008 * scale, 0.0, z_foil],
        length=submergence_depth + 0.15 * scale,
        root_chord=0.085 * scale,
        tip_chord=0.070 * scale,
        airfoil=strut_airfoil,
    )

    if not include_stabilizer:
        pod = hsb.HydroFuselage(
            name="Junction Fairing",
            xsecs=[
                asb.FuselageXSec(xyz_c=[-0.040 * scale, 0.0, z_pod], radius=0.0020 * scale),
                asb.FuselageXSec(xyz_c=[-0.015 * scale, 0.0, z_pod], radius=0.0095 * scale),
                asb.FuselageXSec(xyz_c=[0.020 * scale, 0.0, z_pod], radius=0.0140 * scale),
                asb.FuselageXSec(xyz_c=[0.070 * scale, 0.0, z_pod], radius=0.0135 * scale),
                asb.FuselageXSec(xyz_c=[0.115 * scale, 0.0, z_pod], radius=0.0085 * scale),
                asb.FuselageXSec(xyz_c=[0.155 * scale, 0.0, z_pod], radius=0.0020 * scale),
            ],
        )
        return hsb.HydrofoilAssembly(
            name="Optimized Stern T-Foil",
            xyz_ref=xyz_ref,
            foils=[main_foil],
            struts=[strut],
            fuselages=[pod],
        )

    stab_airfoil = asb.Airfoil("naca0010")
    stabilizer = hsb.Foil(
        name="Horizontal Stabilizer",
        symmetric=True,
        role="stabilizer",
        xsecs=[
            asb.WingXSec(
                xyz_le=[0.48 * scale, 0.0, z_foil],
                chord=0.055 * scale,
                twist=stab_incidence,
                airfoil=stab_airfoil,
            ),
            asb.WingXSec(
                xyz_le=[0.485 * scale, 0.14 * scale, z_foil],
                chord=0.040 * scale,
                twist=stab_incidence,
                airfoil=stab_airfoil,
            ),
        ],
    )
    pod_boom = hsb.HydroFuselage(
        name="Fuselage Pod",
        xsecs=[
            asb.FuselageXSec(xyz_c=[-0.10 * scale, 0.0, z_foil], radius=0.003 * scale),
            asb.FuselageXSec(xyz_c=[0.04 * scale, 0.0, z_foil], radius=0.013 * scale),
            asb.FuselageXSec(xyz_c=[0.32 * scale, 0.0, z_foil], radius=0.011 * scale),
            asb.FuselageXSec(xyz_c=[0.56 * scale, 0.0, z_foil], radius=0.003 * scale),
        ],
    )
    return hsb.HydrofoilAssembly(
        name="Optimized T-Foil Assembly",
        xyz_ref=xyz_ref,
        foils=[main_foil, stabilizer],
        struts=[strut],
        fuselages=[pod_boom],
    )


def run_optimization(
    velocity: float = 6.0,
    craft_mass_kg: float = 30.0,
    submergence_depth: float = 0.35,
    max_aspect_ratio: float = 11.0,
    min_taper_ratio: float = 0.40,
    include_stabilizer: bool = False,
    verbose: bool = False,
) -> dict:
    """Solve the T-foil hydrofoil MDO problem and cross-check all 3D solvers."""
    water = hsb.Water.seawater(temperature_C=18.0)
    scale = (craft_mass_kg / 30.0) ** (1.0 / 3.0)
    xyz_ref = (
        [0.03 * scale, 0.0, -0.75 * submergence_depth]
        if include_stabilizer
        else [0.03 * scale, 0.0, 0.0]
    )

    opti = asb.Opti()

    half_span_main = opti.variable(
        init_guess=0.32 * scale,
        lower_bound=0.15 * scale,
        upper_bound=0.60 * scale,
    )
    chord_root_main = opti.variable(
        init_guess=0.09 * scale,
        lower_bound=0.045 * scale,
        upper_bound=0.22 * scale,
    )
    chord_tip_main = opti.variable(
        init_guess=0.06 * scale,
        lower_bound=0.030 * scale,
        upper_bound=0.18 * scale,
    )

    if include_stabilizer:
        wing_incidence = 0.0
        stab_incidence = opti.variable(init_guess=-2.0, lower_bound=-8.0, upper_bound=4.0)
        alpha = opti.variable(init_guess=3.0, lower_bound=-3.0, upper_bound=10.0)
    else:
        wing_incidence = opti.variable(init_guess=3.0, lower_bound=-3.0, upper_bound=10.0)
        stab_incidence = 0.0
        alpha = 0.0

    assembly = build_hydrofoil_assembly(
        half_span_main=half_span_main,
        chord_root_main=chord_root_main,
        chord_tip_main=chord_tip_main,
        wing_incidence=wing_incidence,
        xyz_ref=xyz_ref,
        submergence_depth=submergence_depth,
        scale=scale,
        include_stabilizer=include_stabilizer,
        stab_incidence=stab_incidence,
    )

    op_point = hsb.HydroOperatingPoint(
        water=water,
        velocity=velocity,
        alpha=alpha,
        surface_z=0.0,
    )
    wave_scales = (
        0.8 * scale,
        (0.65 if include_stabilizer else 0.20) * scale,
        submergence_depth,
    )
    buildup = hsb.HydroBuildup(
        assembly=assembly,
        op_point=op_point,
        model_size="small",
        free_surface_mode="high_froude",
        wave_length_scales=wave_scales,
    )
    res = buildup.run()

    ar_main = (2.0 * half_span_main) / (0.5 * (chord_root_main + chord_tip_main))
    weight_N = craft_mass_kg * water.gravity
    constraints = [
        res["L"] + res["buoyancy_force"] == weight_N,
        res["cavitation_margin"] >= 0.15,
        res["ventilation_margin"] >= 0.10,
        chord_tip_main <= chord_root_main,
        chord_tip_main >= min_taper_ratio * chord_root_main,
        ar_main <= max_aspect_ratio,
    ]

    if include_stabilizer:
        delta_alpha_deg = 0.5
        op_pert = hsb.HydroOperatingPoint(
            water=water,
            velocity=velocity,
            alpha=alpha + delta_alpha_deg,
            surface_z=0.0,
        )
        res_pert = hsb.HydroBuildup(
            assembly=assembly,
            op_point=op_pert,
            model_size="small",
            free_surface_mode="high_froude",
            wave_length_scales=wave_scales,
        ).run()
        cma_approx = (res_pert["Cm"] - res["Cm"]) / np.radians(delta_alpha_deg)
        constraints.extend([res["m_b"] == 0.0, cma_approx <= -0.10])

    opti.subject_to(constraints)
    opti.minimize(res["D"])
    sol = opti.solve(verbose=verbose)

    opt_vars = {
        "half_span_main": float(sol(half_span_main)),
        "span_main": 2.0 * float(sol(half_span_main)),
        "chord_root_main": float(sol(chord_root_main)),
        "chord_tip_main": float(sol(chord_tip_main)),
        "taper_ratio_main": float(sol(chord_tip_main)) / float(sol(chord_root_main)),
        "aspect_ratio_main": float(sol(ar_main)),
        "wing_incidence": float(sol(wing_incidence)) if include_stabilizer is False else 0.0,
        "stab_incidence": float(sol(stab_incidence)) if include_stabilizer else None,
        "alpha": float(sol(alpha)) if include_stabilizer else 0.0,
    }

    opt_assembly = build_hydrofoil_assembly(
        half_span_main=opt_vars["half_span_main"],
        chord_root_main=opt_vars["chord_root_main"],
        chord_tip_main=opt_vars["chord_tip_main"],
        wing_incidence=opt_vars["wing_incidence"],
        xyz_ref=xyz_ref,
        submergence_depth=submergence_depth,
        scale=scale,
        include_stabilizer=include_stabilizer,
        stab_incidence=opt_vars["stab_incidence"] or -2.0,
    )
    opt_op = hsb.HydroOperatingPoint(
        water=water,
        velocity=velocity,
        alpha=opt_vars["alpha"],
        surface_z=0.0,
    )

    buildup_res = hsb.HydroBuildup(
        assembly=opt_assembly,
        op_point=opt_op,
        model_size="small",
        free_surface_mode="high_froude",
        wave_length_scales=wave_scales,
    ).run_with_stability_derivatives()

    vlm_res = hsb.HydroVLM(
        assembly=opt_assembly,
        op_point=opt_op,
        spanwise_resolution=8,
        chordwise_resolution=6,
        free_surface_mode="high_froude",
        clip_to_free_surface=True,
    ).run_with_stability_derivatives()

    ll_res = hsb.HydroLiftingLine(
        assembly=opt_assembly,
        op_point=opt_op,
        spanwise_resolution=6,
        model_size="medium",
        free_surface_mode="high_froude",
        clip_to_free_surface=True,
    ).run_with_stability_derivatives()

    return {
        "velocity": float(velocity),
        "craft_mass_kg": float(craft_mass_kg),
        "submergence_depth": float(submergence_depth),
        "include_stabilizer": bool(include_stabilizer),
        "assembly": opt_assembly,
        "design": opt_vars,
        "weight_N": float(weight_N),
        "buildup": buildup_res,
        "vlm": vlm_res,
        "lifting_line": ll_res,
    }


def _scalar(val) -> float:
    return float(np.asarray(val).reshape(-1)[0])


def _fmt(val) -> str:
    return f"{_scalar(val):8.3f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize a stern lift-assist T-foil hydrofoil assembly for minimum drag."
    )
    parser.add_argument(
        "--velocity", type=float, default=6.0, help="Cruise speed V [m/s] (default: 6.0)"
    )
    parser.add_argument(
        "--mass", type=float, default=30.0, help="Target craft lift mass m [kg] (default: 30.0)"
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=0.35,
        help="Main foil submergence depth h [m] (default: 0.35)",
    )
    parser.add_argument(
        "--max-ar",
        type=float,
        default=11.0,
        help="Maximum main foil aspect ratio (default: 11.0)",
    )
    parser.add_argument(
        "--min-taper",
        type=float,
        default=0.40,
        help="Minimum main foil tip-to-root chord taper ratio (default: 0.40)",
    )
    parser.add_argument(
        "--include-stabilizer",
        action="store_true",
        help="Include an aft horizontal stabilizer and standalone trim constraints.",
    )
    parser.add_argument(
        "--export-step",
        type=str,
        default=None,
        help="Optional path to export optimal geometry as a .step file",
    )
    args = parser.parse_args()

    results = run_optimization(
        velocity=args.velocity,
        craft_mass_kg=args.mass,
        submergence_depth=args.depth,
        max_aspect_ratio=args.max_ar,
        min_taper_ratio=args.min_taper,
        include_stabilizer=args.include_stabilizer,
        verbose=False,
    )
    d = results["design"]
    title = "Optimal T-Foil Assembly (Stabilized)" if results["include_stabilizer"] else "Optimal Stern Lift-Assist T-Foil"
    print(
        f"=== {title} "
        f"(V = {results['velocity']:.1f} m/s, m = {results['craft_mass_kg']:.1f} kg, "
        f"h = {results['submergence_depth']:.2f} m) ==="
    )
    print(f"Main span        : {d['span_main']:.3f} m (half = {d['half_span_main']:.3f} m)")
    print(f"Main root chord  : {d['chord_root_main']:.4f} m")
    print(f"Main tip chord   : {d['chord_tip_main']:.4f} m (taper = {d['taper_ratio_main']:.2f})")
    print(f"Main aspect ratio: {d['aspect_ratio_main']:.2f}")
    print(f"Wing incidence   : {d['wing_incidence']:.3f} deg (flow/mast alpha = {d['alpha']:.2f} deg)")
    if d["stab_incidence"] is not None:
        print(f"Stab incidence   : {d['stab_incidence']:.3f} deg")
    print(f"Target force     : {results['weight_N']:.2f} N")
    print()
    print("Solver Cross-Check:")
    header = (
        f"{'Solver':<18} {'L [N]':>9} {'D [N]':>9} {'P [W]':>9} {'L/D':>8} "
        f"{'CL':>8} {'CD':>8} {'CavMarg':>8} {'V_inc':>8} {'Buoy[N]':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, key in [
        ("HydroBuildup", "buildup"),
        ("HydroVLM", "vlm"),
        ("HydroLiftingLine", "lifting_line"),
    ]:
        r = results[key]
        ld = _scalar(r["L"]) / max(_scalar(r["D"]), 1e-9)
        print(
            f"{name:<18} {_fmt(r['L']):>9} {_fmt(r['D']):>9} {_fmt(r['P_drag']):>9} {ld:8.2f} "
            f"{_fmt(r['CL']):>8} {_fmt(r['CD']):>8} "
            f"{_fmt(r['cavitation_margin']):>8} "
            f"{_fmt(r['V_cavitation_inception']):>8} "
            f"{_fmt(r['buoyancy_force']):>8}"
        )
    b = results["buildup"]
    v = results["velocity"]
    print()
    print(
        f"Drag Breakdown   : D_profile={_scalar(b['D_profile']):.2f} N ({_scalar(b['D_profile']) * v:.1f} W), "
        f"D_induced={_scalar(b['D_induced']):.2f} N ({_scalar(b['D_induced']) * v:.1f} W), "
        f"D_spray={_scalar(b['D_spray']):.2f} N ({_scalar(b['D_spray']) * v:.1f} W), "
        f"D_junction={_scalar(b['D_junction']):.2f} N ({_scalar(b['D_junction']) * v:.1f} W), "
        f"D_wave={_scalar(b['D_wave']):.2f} N ({_scalar(b['D_wave']) * v:.1f} W) | "
        f"Total P_drag={_scalar(b['P_drag']):.1f} W"
    )
    print(
        f"Stern Mount Load : m_b={_scalar(b['m_b']):.2f} N*m (about z=0 mount), "
        f"CLa={_scalar(b['CLa']):.3f}/rad, CLh={_scalar(b['CLh']):.3f}/m, L_h={_scalar(b['L_h']):.1f} N/m"
    )
    if args.export_step:
        step_path = results["assembly"].export_step(args.export_step)
        print(f"Exported STEP geometry to: {step_path}")
