import pytest
import aerosandbox as asb
import aerosandbox.numpy as np
from hydrosandbox.fluid import Water
from hydrosandbox.geometry import (
    Foil,
    HydroFuselage,
    HydrofoilAssembly,
    Strut,
    clip_wing_to_free_surface,
)


def test_assembly_components_and_reference_values():
    """Verify HydrofoilAssembly pulls s_ref/c_ref/b_ref from the primary Foil, not the Strut."""
    main_foil = Foil(
        name="Main Foil",
        symmetric=True,
        role="main",
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, -0.6], chord=0.16, airfoil=asb.Airfoil("naca2412")),
            asb.WingXSec(xyz_le=[0.02, 0.50, -0.6], chord=0.10, airfoil=asb.Airfoil("naca2412")),
        ],
    )
    strut = Strut.from_dimensions(
        name="Mast",
        root_xyz=[0.02, 0.0, -0.6],
        length=1.0,
        root_chord=0.22,
        tip_chord=0.20,
    )
    pod = HydroFuselage(
        name="Pod",
        xsecs=[
            asb.FuselageXSec(xyz_c=[-0.15, 0.0, -0.6], radius=0.005),
            asb.FuselageXSec(xyz_c=[0.30, 0.0, -0.6], radius=0.03),
            asb.FuselageXSec(xyz_c=[0.80, 0.0, -0.6], radius=0.005),
        ],
    )
    assembly = HydrofoilAssembly(
        name="Test Craft",
        xyz_ref=[0.04, 0.0, -0.3],
        foils=[main_foil],
        struts=[strut],
        fuselages=[pod],
    )

    assert len(assembly.foils) == 1
    assert len(assembly.struts) == 1
    assert len(assembly.wings) == 2
    assert np.isclose(assembly.s_ref, main_foil.area(), rtol=1e-9)
    assert np.isclose(assembly.b_ref, main_foil.span(include_centerline_distance=True), rtol=1e-9)
    assert np.isclose(assembly.c_ref, main_foil.mean_aerodynamic_chord(), rtol=1e-9)


def test_archimedes_buoyancy_and_submerged_centroid():
    """Verify displaced volume, center of buoyancy shift, force, and moment."""
    strut = Strut.from_dimensions(
        name="Vertical Strut",
        root_xyz=[0.0, 0.0, -1.0],
        length=1.0,
        root_chord=0.20,
        tip_chord=0.20,
        airfoil=asb.Airfoil("naca0012"),
    )
    assembly = HydrofoilAssembly(
        name="Buoyant Strut",
        xyz_ref=[0.5, 0.0, -0.5],
        struts=[strut],
    )
    water = Water.freshwater(temperature_C=15.0)

    v_full = assembly.displaced_volume(surface_z=0.5)
    v_half = assembly.displaced_volume(surface_z=-0.5)
    v_quarter = assembly.displaced_volume(surface_z=-0.75)

    assert np.isclose(v_full, strut.volume(), rtol=1e-4)
    assert np.isclose(v_half, 0.5 * v_full, rtol=0.02)
    assert np.isclose(v_quarter, 0.25 * v_full, rtol=0.02)

    cb_full = assembly.center_of_buoyancy(surface_z=0.5)
    cb_half = assembly.center_of_buoyancy(surface_z=-0.5)
    assert np.isclose(cb_full[2], -0.5, atol=0.02)
    assert np.isclose(cb_half[2], -0.75, atol=0.02)
    assert cb_half[2] < cb_full[2]

    hydro = assembly.hydrostatic_forces(water=water, surface_z=-0.5, xyz_ref=[0.5, 0.0, -0.5])
    expected_f_z = float(water.density() * water.gravity * v_half)
    assert np.isclose(hydro["buoyancy_force"], expected_f_z, rtol=1e-9)
    assert np.allclose(hydro["F_buoyancy_g"], [0.0, 0.0, expected_f_z], rtol=1e-9)

    # Buoyancy acting at x_cb ~ 0.08 m ahead of xyz_ref[0] = 0.50 m lifts the bow (-M_y < 0 -> wait:
    # r_arm = [x_cb - 0.5 < 0, 0, dz], F = [0, 0, +F_z] -> (r x F)_y = -dx * F_z = +(+0.42) * F_z > 0 -> nose-up!).
    r_arm = np.array(hydro["xyz_cb"]) - np.array([0.5, 0.0, -0.5])
    expected_m = np.cross(r_arm, np.array(hydro["F_buoyancy_g"]))
    assert np.allclose(hydro["M_buoyancy_g"], expected_m, rtol=1e-9)


def test_clip_wing_to_free_surface():
    """Verify exact geometric clipping of piercing, submerged, and emerged wings."""
    strut = Strut.from_dimensions(
        name="Piercing Strut",
        root_xyz=[0.0, 0.0, -0.8],
        length=1.2,
        root_chord=0.20,
        tip_chord=0.08,
        twist_root=0.0,
        twist_tip=1.2,
    )

    clipped = clip_wing_to_free_surface(strut, surface_z=0.0)
    assert len(clipped) == 1
    w_sub = clipped[0]
    assert np.isclose(w_sub.xsecs[0].xyz_le[2], -0.8, atol=1e-9)
    assert np.isclose(w_sub.xsecs[-1].xyz_le[2], 0.0, atol=1e-9)
    # 2/3 of 1.2 m length is submerged -> t = 2/3
    expected_wl_chord = 0.20 + (2.0 / 3.0) * (0.08 - 0.20)
    assert np.isclose(w_sub.xsecs[-1].chord, expected_wl_chord, atol=1e-9)
    assert np.isclose(w_sub.xsecs[-1].twist, 0.8, atol=1e-9)

    assert len(clip_wing_to_free_surface(strut, surface_z=-1.0)) == 0
    assert clip_wing_to_free_surface(strut, surface_z=2.0) == [strut]


def test_export_step_manifold_and_volume(tmp_path):
    """Verify native AP214 STEP export produces manifold watertight B-Rep shells matching volume."""
    import re
    from scipy.interpolate import BSpline

    main_foil = Foil(
        name="Main Foil",
        symmetric=True,
        role="main",
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, -0.35], chord=0.12, airfoil=asb.Airfoil("naca0012")),
            asb.WingXSec(xyz_le=[0.01, 0.36, -0.35], chord=0.06, airfoil=asb.Airfoil("naca0012")),
        ],
    )
    strut = Strut.from_dimensions(
        name="Mast",
        root_xyz=[0.015, 0.0, -0.35],
        length=0.50,
        root_chord=0.09,
        tip_chord=0.07,
        airfoil=asb.Airfoil("naca0010"),
    )
    pod = HydroFuselage(
        name="Pod",
        xsecs=[
            asb.FuselageXSec(xyz_c=[-0.10, 0.0, -0.35], radius=0.004),
            asb.FuselageXSec(xyz_c=[0.10, 0.0, -0.35], radius=0.015),
            asb.FuselageXSec(xyz_c=[0.45, 0.0, -0.35], radius=0.004),
        ],
    )
    assembly = HydrofoilAssembly(
        name="STEP Test Foil",
        foils=[main_foil],
        struts=[strut],
        fuselages=[pod],
    )

    step_path = assembly.export_step(tmp_path / "test_foil.step")
    assert step_path.exists()
    text = step_path.read_text(encoding="utf-8")

    assert "ISO-10303-21;" in text
    assert "AUTOMOTIVE_DESIGN" in text
    assert "B_SPLINE_SURFACE_WITH_KNOTS" in text
    assert "END-ISO-10303-21;" in text

    entities: dict[int, tuple[str, str]] = {}
    for match in re.finditer(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\((.*)\);$", text, flags=re.M):
        entities[int(match.group(1))] = (match.group(2), match.group(3))

    # 1. No bare integer tokens in CARTESIAN_POINT / DIRECTION coordinate triples
    for eid, (etype, args) in entities.items():
        if etype in ("CARTESIAN_POINT", "DIRECTION"):
            nums = re.search(r"\(([^()]+)\)$", args)
            assert nums is not None, f"Entity #{eid} malformed"
            for tok in nums.group(1).split(","):
                assert "." in tok, f"Bare integer {tok!r} in REAL tuple of #{eid}"

    # 2. Validate 3 MANIFOLD_SOLID_BREPs (continuous symmetric wing + mast + pod),
    #    Euler V - E + F = 2, edge parity (+ / -), & volume
    solids = [eid for eid, (et, _) in entities.items() if et == "MANIFOLD_SOLID_BREP"]
    assert len(solids) == 3

    # Also verify unmerged option preserves split Main Foil + Main Foil_mirror solids
    split_path = assembly.export_step(
        tmp_path / "test_foil_split.step", merge_symmetric_wings=False
    )
    split_text = split_path.read_text(encoding="utf-8")
    assert len(re.findall(r"MANIFOLD_SOLID_BREP\(", split_text)) == 4

    def _parse_cartesian(cid: int) -> np.ndarray:
        _, args = entities[cid]
        m = re.search(r"\(([^()]+)\)$", args)
        return np.array([float(x) for x in m.group(1).split(",")], dtype=float)

    total_brep_volume_mm3 = 0.0

    for sid in solids:
        _, sargs = entities[sid]
        shell_id = int(re.findall(r"#(\d+)", sargs)[0])
        _, shell_args = entities[shell_id]
        face_ids = [int(x) for x in re.findall(r"#(\d+)", shell_args)]

        verts_in_solid: set[int] = set()
        edge_senses: dict[int, list[str]] = {}
        solid_vol_mm3 = 0.0

        for fid in face_ids:
            _, fargs = entities[fid]
            refs = [int(x) for x in re.findall(r"#(\d+)", fargs)]
            fob_id, surf_id = refs[0], refs[1]
            _, fob_args = entities[fob_id]
            loop_id = int(re.findall(r"#(\d+)", fob_args)[0])
            _, loop_args = entities[loop_id]
            oe_ids = [int(x) for x in re.findall(r"#(\d+)", loop_args)]

            loop_pts: list[np.ndarray] = []
            for oe_id in oe_ids:
                _, oe_args = entities[oe_id]
                ec_id = int(re.findall(r"#(\d+)", oe_args)[0])
                sense = ".T." if oe_args.endswith(".T.") else ".F."
                edge_senses.setdefault(ec_id, []).append(sense)

                _, ec_args = entities[ec_id]
                v_a, v_b, curve_id = [int(x) for x in re.findall(r"#(\d+)", ec_args)]
                verts_in_solid.update((v_a, v_b))

                _, c_args = entities[curve_id]
                # Sample edge curve densely for planar endcap polygon integration
                deg_m = re.match(r"[^,]*,(\d+),\(([^)]+)\)", c_args)
                deg = int(deg_m.group(1))
                c_cids = [int(x) for x in re.findall(r"#(\d+)", deg_m.group(2))]
                c_cps = np.array([_parse_cartesian(cid) for cid in c_cids])
                tail_tuples = re.findall(r"\(([0-9.,eE+-]+)\)", c_args)
                c_mults = [int(x) for x in tail_tuples[-2].split(",")]
                c_knots = [float(x) for x in tail_tuples[-1].split(",")]
                t_full = np.concatenate(
                    [np.full(m, k) for k, m in zip(c_knots, c_mults)]
                )
                bs = BSpline(t_full, c_cps, k=deg)
                s_pts = bs(np.linspace(0.0, 1.0, 16 if deg > 1 else 2))
                if sense == ".F.":
                    s_pts = s_pts[::-1]
                loop_pts.extend(s_pts[:-1])

            stype, s_args = entities[surf_id]
            if stype == "PLANE":
                ax_id = int(re.findall(r"#(\d+)", s_args)[0])
                _, ax_args = entities[ax_id]
                orig_cid, _, _ = [int(x) for x in re.findall(r"#(\d+)", ax_args)]
                center = _parse_cartesian(orig_cid)
                for k in range(len(loop_pts)):
                    p_a = loop_pts[k]
                    p_b = loop_pts[(k + 1) % len(loop_pts)]
                    solid_vol_mm3 += float(np.dot(np.cross(center, p_a), p_b)) / 6.0
            else:
                # Parse B_SPLINE_SURFACE_WITH_KNOTS for direct NURBS volume check
                head_m = re.match(r"[^,]*,(\d+),(\d+),(\(\([^;]+\)\)),\.UNSPECIFIED\.", s_args)
                u_deg, v_deg = int(head_m.group(1)), int(head_m.group(2))
                rows = re.findall(r"\((#[0-9,#]+)\)", head_m.group(3))
                cp_grid = np.array(
                    [[_parse_cartesian(int(x)) for x in re.findall(r"#(\d+)", r)] for r in rows]
                )
                rest = s_args[head_m.end():]
                toks = re.findall(r"\(([0-9.,eE+-]+)\)", rest)
                u_mults = [int(x) for x in toks[0].split(",")]
                v_mults = [int(x) for x in toks[1].split(",")]
                u_knots = [float(x) for x in toks[2].split(",")]
                v_knots = [float(x) for x in toks[3].split(",")]
                tu = np.concatenate([np.full(m, k) for k, m in zip(u_knots, u_mults)])
                tv = np.concatenate([np.full(m, k) for k, m in zip(v_knots, v_mults)])

                # Evaluate tensor-product spline on (u, v) = (32, 2)
                us = np.linspace(0.0, 1.0, 32)
                vs = np.linspace(0.0, 1.0, 2)
                bu = BSpline.design_matrix(us, tu, u_deg).toarray()
                bv = BSpline.design_matrix(vs, tv, v_deg).toarray()
                surf_pts = np.einsum("iu,uvd,jv->ijd", bu, cp_grid, bv)
                for ii in range(len(us) - 1):
                    for jj in range(len(vs) - 1):
                        p00 = surf_pts[ii, jj]
                        p10 = surf_pts[ii + 1, jj]
                        p11 = surf_pts[ii + 1, jj + 1]
                        p01 = surf_pts[ii, jj + 1]
                        solid_vol_mm3 += float(np.dot(np.cross(p00, p10), p11)) / 6.0
                        solid_vol_mm3 += float(np.dot(np.cross(p00, p11), p01)) / 6.0

        # Check Euler characteristic V - E + F = 2 and +1 / -1 edge parity
        assert len(verts_in_solid) - len(edge_senses) + len(face_ids) == 2
        for ec_id, senses in edge_senses.items():
            assert sorted(senses) == [".F.", ".T."], f"Edge #{ec_id} non-manifold senses {senses}"

        assert solid_vol_mm3 > 0.0
        total_brep_volume_mm3 += solid_vol_mm3

    expected_vol_mm3 = float(assembly.displaced_volume(surface_z=None)) * 1e9
    assert np.isclose(total_brep_volume_mm3, expected_vol_mm3, rtol=0.03)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
