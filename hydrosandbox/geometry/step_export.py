import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from aerosandbox.geometry import Wing
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline


def _fmt_real(value: float) -> str:
    """Format a floating-point number as an ISO 10303-21 REAL literal (always contains '.')."""
    v = float(value)
    if abs(v) < 1e-14:
        return "0."
    s = f"{v:.10g}"
    if "." not in s:
        if "e" in s:
            mantissa, exp = s.split("e", 1)
            s = f"{mantissa}.e{exp}"
        elif "E" in s:
            mantissa, exp = s.split("E", 1)
            s = f"{mantissa}.E{exp}"
        else:
            s = f"{s}."
    return s


def _knot_multiplicities(t: np.ndarray, tol: float = 1e-12) -> tuple[list[float], list[int]]:
    """Convert a clamped B-spline knot array into distinct knots and multiplicities."""
    knots: list[float] = [float(t[0])]
    mults: list[int] = [1]
    for k in t[1:]:
        kf = float(k)
        if abs(kf - knots[-1]) <= tol:
            mults[-1] += 1
        else:
            knots.append(kf)
            mults.append(1)
    knots[0] = 0.0
    knots[-1] = 1.0
    return knots, mults


def _resample_curve_2d(xy: np.ndarray, s_eval: np.ndarray) -> np.ndarray:
    """Resample a 2D polyline onto normalized arc-length coordinates ``s_eval`` in ``[0, 1]``."""
    diffs = np.diff(xy, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(ds)])
    total = float(arc[-1])
    if total <= 0.0:
        return np.repeat(xy[:1], len(s_eval), axis=0)
    arc_norm = arc / total
    x_new = np.interp(s_eval, arc_norm, xy[:, 0])
    y_new = np.interp(s_eval, arc_norm, xy[:, 1])
    return np.column_stack([x_new, y_new])


def _fit_clamped_cubic_3d(s: np.ndarray, pts_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a clamped cubic B-spline through ``pts_3d`` preserving exact endpoints."""
    bspl = make_interp_spline(s, pts_3d, k=3)
    cps = np.array(bspl.c, dtype=float)
    cps[0] = pts_3d[0]
    cps[-1] = pts_3d[-1]
    return cps, np.array(bspl.t, dtype=float)


def _triangle_signed_volume(triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """Compute signed volume from oriented triangles using divergence theorem: (1/6) sum (a x b) . c."""
    vol = 0.0
    for a, b, c in triangles:
        vol += float(np.dot(np.cross(a, b), c))
    return vol / 6.0


def _grid_triangles(grid_pts: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Triangulate an ``(Nu, Nv, 3)`` surface grid with orientation matching ``dS/du x dS/dv``.
    """
    nu, nv, _ = grid_pts.shape
    tris: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            p00 = grid_pts[i, j]
            p10 = grid_pts[i + 1, j]
            p11 = grid_pts[i + 1, j + 1]
            p01 = grid_pts[i, j + 1]
            tris.append((p00, p10, p11))
            tris.append((p00, p11, p01))
    return tris


def _polygon_normal_and_triangles(
    center: np.ndarray, boundary_pts: np.ndarray
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Return the unit CCW polygon normal and fan triangles ``(center, p_k, p_{k+1})``
    for an ordered closed boundary polyline ``boundary_pts`` (without closing duplicate).
    """
    tris: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    area_vec = np.zeros(3, dtype=float)
    n_pts = len(boundary_pts)
    for k in range(n_pts):
        p_a = boundary_pts[k]
        p_b = boundary_pts[(k + 1) % n_pts]
        cross = np.cross(p_a - center, p_b - center)
        area_vec += cross
        tris.append((center, p_a, p_b))
    norm = float(np.linalg.norm(area_vec))
    normal = area_vec / max(norm, 1e-16)
    return normal, tris


@dataclass
class _SolidEdge:
    v_start: int
    v_end: int
    cp_ids: list[int]
    degree: int
    knots: list[float]
    mults: list[int]


@dataclass
class _SolidFace:
    kind: str  # "bspline" or "plane"
    loop: list[tuple[int, bool]]  # (edge_idx, forward_sense) CCW against natural normal
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    # B-spline surface fields
    u_degree: int = 1
    v_degree: int = 1
    cp_grid_ids: list[list[int]] | None = None
    u_knots: list[float] | None = None
    u_mults: list[int] | None = None
    v_knots: list[float] | None = None
    v_mults: list[int] | None = None
    # Plane fields
    plane_origin_id: int | None = None
    plane_normal: np.ndarray | None = None
    plane_ref_dir: np.ndarray | None = None


class _StepWriter:
    """Low-level ISO 10303-21 Part 21 entity writer."""

    def __init__(self, product_name: str):
        self.product_name = product_name
        self._next_id = 1
        self._lines: list[str] = []

    def add(self, body: str) -> int:
        eid = self._next_id
        self._next_id += 1
        self._lines.append(f"#{eid}={body};")
        return eid

    def cartesian_point(self, xyz: np.ndarray | tuple[float, float, float], name: str = "") -> int:
        return self.add(
            f"CARTESIAN_POINT('{name}',({_fmt_real(xyz[0])},{_fmt_real(xyz[1])},{_fmt_real(xyz[2])}))"
        )

    def direction(self, dxyz: np.ndarray | tuple[float, float, float], name: str = "") -> int:
        v = np.asarray(dxyz, dtype=float)
        n = float(np.linalg.norm(v))
        if n > 0.0:
            v = v / n
        return self.add(
            f"DIRECTION('{name}',({_fmt_real(v[0])},{_fmt_real(v[1])},{_fmt_real(v[2])}))"
        )

    def axis2_placement_3d(
        self,
        origin_id: int,
        normal: np.ndarray | tuple[float, float, float],
        ref_dir: np.ndarray | tuple[float, float, float],
        name: str = "",
    ) -> int:
        n_vec = np.asarray(normal, dtype=float)
        n_vec = n_vec / max(float(np.linalg.norm(n_vec)), 1e-16)
        r_vec = np.asarray(ref_dir, dtype=float)
        # Re-orthogonalize ref_dir against normal to ensure strict orthogonality in CAD readers.
        r_vec = r_vec - np.dot(r_vec, n_vec) * n_vec
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm < 1e-12:
            candidate = np.array([1.0, 0.0, 0.0]) if abs(n_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            r_vec = np.cross(n_vec, candidate)
            r_norm = float(np.linalg.norm(r_vec))
        r_vec = r_vec / r_norm

        z_id = self.direction(n_vec)
        x_id = self.direction(r_vec)
        return self.add(f"AXIS2_PLACEMENT_3D('{name}',#{origin_id},#{z_id},#{x_id})")

    def emit_solid(self, name: str, vertices_xyz: list[np.ndarray], edges: list[_SolidEdge], faces: list[_SolidFace]) -> tuple[int, float]:
        """Orient faces outward, validate manifoldness & Euler characteristic, and emit STEP solid."""
        n_faces = len(faces)
        edge_to_faces: dict[int, list[tuple[int, bool]]] = {i: [] for i in range(len(edges))}
        for f_idx, face in enumerate(faces):
            for e_idx, fwd in face.loop:
                edge_to_faces[e_idx].append((f_idx, fwd))

        for e_idx, uses in edge_to_faces.items():
            if len(uses) != 2:
                raise ValueError(
                    f"Non-manifold solid '{name}': edge {e_idx} has {len(uses)} incident faces (expected 2)."
                )

        # Propagate consistent relative orientations via BFS from face 0
        flips: list[bool | None] = [None] * n_faces
        flips[0] = False
        queue = [0]
        while queue:
            curr = queue.pop(0)
            curr_flip = bool(flips[curr])
            for e_idx, fwd_curr in faces[curr].loop:
                uses = edge_to_faces[e_idx]
                other_idx, fwd_other = uses[1] if uses[0][0] == curr else uses[0]
                # Edge must be traversed in opposite directions on the two adjacent faces
                needed_flip = curr_flip ^ (fwd_curr == fwd_other)
                if flips[other_idx] is None:
                    flips[other_idx] = needed_flip
                    queue.append(other_idx)
                elif flips[other_idx] != needed_flip:
                    raise ValueError(
                        f"Non-orientable shell on solid '{name}' at edge {e_idx}."
                    )

        if any(f is None for f in flips):
            raise ValueError(f"Disconnected shell components in solid '{name}'.")

        face_flips = [bool(f) for f in flips]

        # Ensure outward orientation via signed volume
        signed_vol = 0.0
        for f_idx, face in enumerate(faces):
            v_face = _triangle_signed_volume(face.triangles)
            signed_vol += -v_face if face_flips[f_idx] else v_face

        if signed_vol < 0.0:
            face_flips = [not f for f in face_flips]
            signed_vol = -signed_vol

        # Verify edge parity after outward orientation
        oriented_counts: dict[int, list[bool]] = {i: [] for i in range(len(edges))}
        for f_idx, face in enumerate(faces):
            flip = face_flips[f_idx]
            for e_idx, fwd in face.loop:
                oriented_counts[e_idx].append(fwd ^ flip)
        for e_idx, senses in oriented_counts.items():
            if set(senses) != {True, False}:
                raise ValueError(f"Solid '{name}' failed edge orientation parity on edge {e_idx}.")

        # Check Euler characteristic V - E + F = 2
        if len(vertices_xyz) - len(edges) + len(faces) != 2:
            raise ValueError(
                f"Solid '{name}' failed closed-shell Euler check: "
                f"V={len(vertices_xyz)}, E={len(edges)}, F={len(faces)}."
            )

        # Emit vertices and EDGE_CURVEs
        vertex_point_ids = []
        vertex_cp_ids = []
        for xyz in vertices_xyz:
            cp_id = self.cartesian_point(xyz)
            vp_id = self.add(f"VERTEX_POINT('',#{cp_id})")
            vertex_cp_ids.append(cp_id)
            vertex_point_ids.append(vp_id)

        edge_curve_ids = []
        for edge in edges:
            cp_refs = ",".join(f"#{cid}" for cid in edge.cp_ids)
            mult_str = ",".join(str(int(m)) for m in edge.mults)
            knot_str = ",".join(_fmt_real(k) for k in edge.knots)
            curve_id = self.add(
                f"B_SPLINE_CURVE_WITH_KNOTS('',{edge.degree},({cp_refs}),"
                f".UNSPECIFIED.,.F.,.F.,({mult_str}),({knot_str}),.UNSPECIFIED.)"
            )
            ec_id = self.add(
                f"EDGE_CURVE('',#{vertex_point_ids[edge.v_start]},#{vertex_point_ids[edge.v_end]},#{curve_id},.T.)"
            )
            edge_curve_ids.append(ec_id)

        # Emit ADVANCED_FACEs (always same_sense = .T. by flipping geometry cleanly when needed)
        face_ids = []
        for f_idx, face in enumerate(faces):
            flip = face_flips[f_idx]
            loop_seq = (
                [(e_idx, not fwd) for e_idx, fwd in reversed(face.loop)]
                if flip
                else list(face.loop)
            )
            oe_ids = []
            for e_idx, fwd in loop_seq:
                sense_tok = ".T." if fwd else ".F."
                oe_ids.append(self.add(f"ORIENTED_EDGE('',*,*,#{edge_curve_ids[e_idx]},{sense_tok})"))
            oe_refs = ",".join(f"#{oid}" for oid in oe_ids)
            el_id = self.add(f"EDGE_LOOP('',({oe_refs}))")
            fob_id = self.add(f"FACE_OUTER_BOUND('',#{el_id},.T.)")

            if face.kind == "plane":
                assert face.plane_origin_id is not None
                assert face.plane_normal is not None
                assert face.plane_ref_dir is not None
                normal = -face.plane_normal if flip else face.plane_normal
                ax_id = self.axis2_placement_3d(
                    face.plane_origin_id, normal, face.plane_ref_dir
                )
                surf_id = self.add(f"PLANE('',#{ax_id})")
            elif face.kind == "bspline":
                assert face.cp_grid_ids is not None
                assert face.u_knots is not None and face.u_mults is not None
                assert face.v_knots is not None and face.v_mults is not None
                if flip:
                    grid_ids = [list(reversed(row)) for row in face.cp_grid_ids]
                    v_knots = [1.0 - vk for vk in reversed(face.v_knots)]
                    v_mults = list(reversed(face.v_mults))
                else:
                    grid_ids = face.cp_grid_ids
                    v_knots = face.v_knots
                    v_mults = face.v_mults

                rows_str = ",".join(
                    "(" + ",".join(f"#{cid}" for cid in row) + ")" for row in grid_ids
                )
                u_mult_str = ",".join(str(int(m)) for m in face.u_mults)
                v_mult_str = ",".join(str(int(m)) for m in v_mults)
                u_knot_str = ",".join(_fmt_real(k) for k in face.u_knots)
                v_knot_str = ",".join(_fmt_real(k) for k in v_knots)
                surf_id = self.add(
                    f"B_SPLINE_SURFACE_WITH_KNOTS('',{face.u_degree},{face.v_degree},"
                    f"({rows_str}),.UNSPECIFIED.,.F.,.F.,.F.,"
                    f"({u_mult_str}),({v_mult_str}),({u_knot_str}),({v_knot_str}),.UNSPECIFIED.)"
                )
            else:
                raise ValueError(f"Unknown surface kind {face.kind!r}")

            af_id = self.add(f"ADVANCED_FACE('',(#{fob_id}),#{surf_id},.T.)")
            face_ids.append(af_id)

        face_refs = ",".join(f"#{fid}" for fid in face_ids)
        shell_id = self.add(f"CLOSED_SHELL('',({face_refs}))")
        safe_name = name.replace("'", "")
        solid_id = self.add(f"MANIFOLD_SOLID_BREP('{safe_name}',#{shell_id})")
        return solid_id, signed_vol

    def build_step_text(self, solid_ids: list[int]) -> str:
        """Assemble the full ISO 10303-21 AP214 STEP file string."""
        safe_prod = self.product_name.replace("'", "")
        origin_id = self.cartesian_point((0.0, 0.0, 0.0))
        ax_id = self.axis2_placement_3d(origin_id, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

        len_unit_id = self.add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
        ang_unit_id = self.add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
        solid_ang_id = self.add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
        uncert_id = self.add(
            f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#{len_unit_id},"
            f"'distance_accuracy_value','confusion accuracy')"
        )
        geom_ctx_id = self.add(
            f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
            f"GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{uncert_id})) "
            f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{len_unit_id},#{ang_unit_id},#{solid_ang_id})) "
            f"REPRESENTATION_CONTEXT('Context #1','3D Context with UNIT and UNCERTAINTY') )"
        )

        items_str = ",".join([f"#{sid}" for sid in solid_ids] + [f"#{ax_id}"])
        shape_rep_id = self.add(
            f"ADVANCED_BREP_SHAPE_REPRESENTATION('{safe_prod}',({items_str}),#{geom_ctx_id})"
        )

        app_ctx_id = self.add(
            "APPLICATION_CONTEXT('configuration controlled 3d designs of mechanical parts and assemblies')"
        )
        self.add(
            f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app_ctx_id})"
        )
        prod_ctx_id = self.add(f"PRODUCT_CONTEXT('',# {app_ctx_id},'mechanical')".replace("# ", "#"))
        prod_id = self.add(f"PRODUCT('{safe_prod}','{safe_prod}','',(#{prod_ctx_id}))")
        self.add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{prod_id}))")
        pdf_id = self.add(f"PRODUCT_DEFINITION_FORMATION('','',#{prod_id})")
        pdc_id = self.add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app_ctx_id},'design')")
        pd_id = self.add(f"PRODUCT_DEFINITION('design','',#{pdf_id},#{pdc_id})")
        pds_id = self.add(f"PRODUCT_DEFINITION_SHAPE('','',#{pd_id})")
        self.add(f"SHAPE_DEFINITION_REPRESENTATION(#{pds_id},#{shape_rep_id})")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        header = (
            "ISO-10303-21;\n"
            "HEADER;\n"
            "FILE_DESCRIPTION(('HydroSandbox NURBS B-Rep Export'),'2;1');\n"
            f"FILE_NAME('{safe_prod}.step','{timestamp}',('HydroSandbox'),('HydroSandbox'),"
            "'HydroSandbox Native STEP Exporter','HydroSandbox','');\n"
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
            "ENDSEC;\n"
            "DATA;\n"
        )
        body = "\n".join(self._lines)
        footer = "\nENDSEC;\nEND-ISO-10303-21;\n"
        return header + body + footer


def _add_wing_solid(
    writer: _StepWriter,
    wing: Any,
    solid_name: str,
    mirror_xz: bool,
    minimum_airfoil_TE_thickness: float,
    n_chordwise: int,
) -> tuple[int, float]:
    """Build NURBS B-Rep surfaces/edges/vertices for one wing (or mirrored side) and emit it."""
    s_eval = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_chordwise)))
    s_dense = np.linspace(0.0, 1.0, 32)
    n_xsecs = len(wing.xsecs)

    vertices_xyz: list[np.ndarray] = []
    edges: list[_SolidEdge] = []
    faces: list[_SolidFace] = []

    v_le_idx: list[int] = []
    v_te_up_idx: list[int] = []
    v_te_lo_idx: list[int] = []

    up_cp_ids_sec: list[list[int]] = []
    lo_cp_ids_sec: list[list[int]] = []
    up_samples_sec: list[np.ndarray] = []
    lo_samples_sec: list[np.ndarray] = []
    pt_le_sec: list[np.ndarray] = []
    pt_te_up_sec: list[np.ndarray] = []
    pt_te_lo_sec: list[np.ndarray] = []

    e_up_idx: list[int] = []
    e_lo_idx: list[int] = []
    e_te_idx: list[int] = []

    u_knots_shared: list[float] | None = None
    u_mults_shared: list[int] | None = None

    for i, xsec in enumerate(wing.xsecs):
        af = xsec.airfoil
        if af.TE_thickness() < minimum_airfoil_TE_thickness:
            af = af.set_TE_thickness(thickness=minimum_airfoil_TE_thickness)

        le_index = af.LE_index()
        xy_upper = np.array(af.coordinates[: le_index + 1, :][::-1], dtype=float)
        xy_lower = np.array(af.coordinates[le_index:, :], dtype=float)

        upper_resamp = _resample_curve_2d(xy_upper, s_eval)
        lower_resamp = _resample_curve_2d(xy_lower, s_eval)

        xg_local, _, zg_local = wing._compute_frame_of_WingXSec(i)
        origin_m = np.array(xsec.xyz_le, dtype=float)
        xg = np.array(xg_local, dtype=float)
        zg = np.array(zg_local, dtype=float)
        chord_f = float(xsec.chord)

        if mirror_xz:
            origin_m[1] *= -1.0
            xg[1] *= -1.0
            zg[1] *= -1.0

        pts_up_mm = 1000.0 * (
            origin_m[None, :]
            + chord_f * (upper_resamp[:, 0:1] * xg[None, :] + upper_resamp[:, 1:2] * zg[None, :])
        )
        pts_lo_mm = 1000.0 * (
            origin_m[None, :]
            + chord_f * (lower_resamp[:, 0:1] * xg[None, :] + lower_resamp[:, 1:2] * zg[None, :])
        )
        # Ensure exact LE junction coincidence
        pts_lo_mm[0] = pts_up_mm[0]

        cps_up, t_u = _fit_clamped_cubic_3d(s_eval, pts_up_mm)
        cps_lo, _ = _fit_clamped_cubic_3d(s_eval, pts_lo_mm)

        if u_knots_shared is None:
            u_knots_shared, u_mults_shared = _knot_multiplicities(t_u)

        up_bspl = BSpline(t_u, cps_up, k=3)
        lo_bspl = BSpline(t_u, cps_lo, k=3)
        up_samples_sec.append(up_bspl(s_dense))
        lo_samples_sec.append(lo_bspl(s_dense))

        pt_le = cps_up[0].copy()
        pt_te_up = cps_up[-1].copy()
        pt_te_lo = cps_lo[-1].copy()
        pt_le_sec.append(pt_le)
        pt_te_up_sec.append(pt_te_up)
        pt_te_lo_sec.append(pt_te_lo)

        il = len(vertices_xyz)
        vertices_xyz.append(pt_le)
        iu = len(vertices_xyz)
        vertices_xyz.append(pt_te_up)
        ilo = len(vertices_xyz)
        vertices_xyz.append(pt_te_lo)
        v_le_idx.append(il)
        v_te_up_idx.append(iu)
        v_te_lo_idx.append(ilo)

        # Emit CARTESIAN_POINTs for B-spline control points (reused on edges and skin surfaces!)
        up_cids = [writer.cartesian_point(pt) for pt in cps_up]
        lo_cids = [writer.cartesian_point(pt) for pt in cps_lo]
        lo_cids[0] = up_cids[0]  # share exact LE CARTESIAN_POINT
        up_cp_ids_sec.append(up_cids)
        lo_cp_ids_sec.append(lo_cids)

        assert u_knots_shared is not None and u_mults_shared is not None
        e_up_idx.append(
            len(edges)
        )
        edges.append(
            _SolidEdge(
                v_start=il,
                v_end=iu,
                cp_ids=up_cids,
                degree=3,
                knots=u_knots_shared,
                mults=u_mults_shared,
            )
        )
        e_lo_idx.append(len(edges))
        edges.append(
            _SolidEdge(
                v_start=il,
                v_end=ilo,
                cp_ids=lo_cids,
                degree=3,
                knots=u_knots_shared,
                mults=u_mults_shared,
            )
        )
        e_te_idx.append(len(edges))
        edges.append(
            _SolidEdge(
                v_start=ilo,
                v_end=iu,
                cp_ids=[lo_cids[-1], up_cids[-1]],
                degree=1,
                knots=[0.0, 1.0],
                mults=[2, 2],
            )
        )

    assert u_knots_shared is not None and u_mults_shared is not None

    # Spanwise edges and ruled loft faces per bay
    for i in range(n_xsecs - 1):
        e_span_le = len(edges)
        edges.append(
            _SolidEdge(
                v_start=v_le_idx[i],
                v_end=v_le_idx[i + 1],
                cp_ids=[up_cp_ids_sec[i][0], up_cp_ids_sec[i + 1][0]],
                degree=1,
                knots=[0.0, 1.0],
                mults=[2, 2],
            )
        )
        e_span_te_up = len(edges)
        edges.append(
            _SolidEdge(
                v_start=v_te_up_idx[i],
                v_end=v_te_up_idx[i + 1],
                cp_ids=[up_cp_ids_sec[i][-1], up_cp_ids_sec[i + 1][-1]],
                degree=1,
                knots=[0.0, 1.0],
                mults=[2, 2],
            )
        )
        e_span_te_lo = len(edges)
        edges.append(
            _SolidEdge(
                v_start=v_te_lo_idx[i],
                v_end=v_te_lo_idx[i + 1],
                cp_ids=[lo_cp_ids_sec[i][-1], lo_cp_ids_sec[i + 1][-1]],
                degree=1,
                knots=[0.0, 1.0],
                mults=[2, 2],
            )
        )

        # Upper skin surface (u: LE -> TE_up, v: sec i -> sec i+1)
        up_grid_ids = [[a, b] for a, b in zip(up_cp_ids_sec[i], up_cp_ids_sec[i + 1])]
        up_grid_pts = np.stack([up_samples_sec[i], up_samples_sec[i + 1]], axis=1)
        faces.append(
            _SolidFace(
                kind="bspline",
                loop=[
                    (e_up_idx[i], True),
                    (e_span_te_up, True),
                    (e_up_idx[i + 1], False),
                    (e_span_le, False),
                ],
                triangles=_grid_triangles(up_grid_pts),
                u_degree=3,
                v_degree=1,
                cp_grid_ids=up_grid_ids,
                u_knots=u_knots_shared,
                u_mults=u_mults_shared,
                v_knots=[0.0, 1.0],
                v_mults=[2, 2],
            )
        )

        # Lower skin surface (u: LE -> TE_lo, v: sec i -> sec i+1)
        lo_grid_ids = [[a, b] for a, b in zip(lo_cp_ids_sec[i], lo_cp_ids_sec[i + 1])]
        lo_grid_pts = np.stack([lo_samples_sec[i], lo_samples_sec[i + 1]], axis=1)
        faces.append(
            _SolidFace(
                kind="bspline",
                loop=[
                    (e_lo_idx[i], True),
                    (e_span_te_lo, True),
                    (e_lo_idx[i + 1], False),
                    (e_span_le, False),
                ],
                triangles=_grid_triangles(lo_grid_pts),
                u_degree=3,
                v_degree=1,
                cp_grid_ids=lo_grid_ids,
                u_knots=u_knots_shared,
                u_mults=u_mults_shared,
                v_knots=[0.0, 1.0],
                v_mults=[2, 2],
            )
        )

        # Trailing-edge flat strip (u: TE_lo -> TE_up, v: sec i -> sec i+1)
        te_grid_ids = [
            [lo_cp_ids_sec[i][-1], lo_cp_ids_sec[i + 1][-1]],
            [up_cp_ids_sec[i][-1], up_cp_ids_sec[i + 1][-1]],
        ]
        te_grid_pts = np.array(
            [
                [pt_te_lo_sec[i], pt_te_lo_sec[i + 1]],
                [pt_te_up_sec[i], pt_te_up_sec[i + 1]],
            ],
            dtype=float,
        )
        faces.append(
            _SolidFace(
                kind="bspline",
                loop=[
                    (e_te_idx[i], True),
                    (e_span_te_up, True),
                    (e_te_idx[i + 1], False),
                    (e_span_te_lo, False),
                ],
                triangles=_grid_triangles(te_grid_pts),
                u_degree=1,
                v_degree=1,
                cp_grid_ids=te_grid_ids,
                u_knots=[0.0, 1.0],
                u_mults=[2, 2],
                v_knots=[0.0, 1.0],
                v_mults=[2, 2],
            )
        )

    # Planar root (i=0) and tip (i=n_xsecs-1) endcaps
    for sec_idx in (0, n_xsecs - 1):
        # Loop (+e_up, -e_te, -e_lo) traces LE -> upper TE -> lower TE -> LE
        poly_pts = np.concatenate(
            [up_samples_sec[sec_idx][:-1], lo_samples_sec[sec_idx][::-1][:-1]], axis=0
        )
        center = 0.5 * (pt_le_sec[sec_idx] + 0.5 * (pt_te_up_sec[sec_idx] + pt_te_lo_sec[sec_idx]))
        n_poly, cap_tris = _polygon_normal_and_triangles(center, poly_pts)
        xg_ref = pt_te_up_sec[sec_idx] - pt_le_sec[sec_idx]
        faces.append(
            _SolidFace(
                kind="plane",
                loop=[
                    (e_up_idx[sec_idx], True),
                    (e_te_idx[sec_idx], False),
                    (e_lo_idx[sec_idx], False),
                ],
                triangles=cap_tris,
                plane_origin_id=up_cp_ids_sec[sec_idx][0],
                plane_normal=n_poly,
                plane_ref_dir=xg_ref,
            )
        )

    return writer.emit_solid(solid_name, vertices_xyz, edges, faces)


def _add_fuselage_solid(
    writer: _StepWriter,
    fuselage: Any,
    solid_name: str,
    fuselage_tol: float,
) -> tuple[int, float]:
    """Build NURBS B-Rep surfaces/edges/vertices for a fuselage pod and emit it."""
    s_param = np.linspace(0.0, 1.0, 25)
    s_dense = np.linspace(0.0, 1.0, 32)
    theta_top = np.pi * s_param
    theta_bot = np.pi + np.pi * s_param

    n_xsecs = len(fuselage.xsecs)
    vertices_xyz: list[np.ndarray] = []
    edges: list[_SolidEdge] = []
    faces: list[_SolidFace] = []

    v_stbd_idx: list[int] = []
    v_port_idx: list[int] = []
    top_cp_ids_sec: list[list[int]] = []
    bot_cp_ids_sec: list[list[int]] = []
    top_samples_sec: list[np.ndarray] = []
    bot_samples_sec: list[np.ndarray] = []
    e_top_idx: list[int] = []
    e_bot_idx: list[int] = []
    centers_mm: list[np.ndarray] = []

    u_knots_shared: list[float] | None = None
    u_mults_shared: list[int] | None = None

    for xsec in fuselage.xsecs:
        if xsec.height < fuselage_tol or xsec.width < fuselage_tol:
            xsec = copy.deepcopy(xsec)
            xsec.height = max(float(xsec.height), fuselage_tol)
            xsec.width = max(float(xsec.width), fuselage_tol)

        xt, yt, zt = xsec.get_3D_coordinates(theta=theta_top)
        xb, yb, zb = xsec.get_3D_coordinates(theta=theta_bot)
        pts_top_mm = 1000.0 * np.column_stack([xt, yt, zt])
        pts_bot_mm = 1000.0 * np.column_stack([xb, yb, zb])

        # Enforce exact shared seam vertices between top & bottom halves
        pts_bot_mm[0] = pts_top_mm[-1]
        pts_bot_mm[-1] = pts_top_mm[0]

        cps_top, t_u = _fit_clamped_cubic_3d(s_param, pts_top_mm)
        cps_bot, _ = _fit_clamped_cubic_3d(s_param, pts_bot_mm)
        if u_knots_shared is None:
            u_knots_shared, u_mults_shared = _knot_multiplicities(t_u)

        top_samples_sec.append(BSpline(t_u, cps_top, k=3)(s_dense))
        bot_samples_sec.append(BSpline(t_u, cps_bot, k=3)(s_dense))
        centers_mm.append(1000.0 * np.array(xsec.xyz_c, dtype=float))

        is_idx = len(vertices_xyz)
        vertices_xyz.append(cps_top[0].copy())
        ip_idx = len(vertices_xyz)
        vertices_xyz.append(cps_top[-1].copy())
        v_stbd_idx.append(is_idx)
        v_port_idx.append(ip_idx)

        top_cids = [writer.cartesian_point(pt) for pt in cps_top]
        bot_cids = [writer.cartesian_point(pt) for pt in cps_bot]
        bot_cids[0] = top_cids[-1]
        bot_cids[-1] = top_cids[0]
        top_cp_ids_sec.append(top_cids)
        bot_cp_ids_sec.append(bot_cids)

        assert u_knots_shared is not None and u_mults_shared is not None
        e_top_idx.append(len(edges))
        edges.append(
            _SolidEdge(
                v_start=is_idx,
                v_end=ip_idx,
                cp_ids=top_cids,
                degree=3,
                knots=u_knots_shared,
                mults=u_mults_shared,
            )
        )
        e_bot_idx.append(len(edges))
        edges.append(
            _SolidEdge(
                v_start=ip_idx,
                v_end=is_idx,
                cp_ids=bot_cids,
                degree=3,
                knots=u_knots_shared,
                mults=u_mults_shared,
            )
        )

    assert u_knots_shared is not None and u_mults_shared is not None

    for i in range(n_xsecs - 1):
        e_ax_stbd = len(edges)
        edges.append(
            _SolidEdge(
                v_start=v_stbd_idx[i],
                v_end=v_stbd_idx[i + 1],
                cp_ids=[top_cp_ids_sec[i][0], top_cp_ids_sec[i + 1][0]],
                degree=1,
                knots=[0.0, 1.0],
                mults=[2, 2],
            )
        )
        e_ax_port = len(edges)
        edges.append(
            _SolidEdge(
                v_start=v_port_idx[i],
                v_end=v_port_idx[i + 1],
                cp_ids=[top_cp_ids_sec[i][-1], top_cp_ids_sec[i + 1][-1]],
                degree=1,
                knots=[0.0, 1.0],
                mults=[2, 2],
            )
        )

        # Top shell (u: stbd -> port via +z, v: sec i -> sec i+1)
        top_grid_ids = [[a, b] for a, b in zip(top_cp_ids_sec[i], top_cp_ids_sec[i + 1])]
        top_grid_pts = np.stack([top_samples_sec[i], top_samples_sec[i + 1]], axis=1)
        faces.append(
            _SolidFace(
                kind="bspline",
                loop=[
                    (e_top_idx[i], True),
                    (e_ax_port, True),
                    (e_top_idx[i + 1], False),
                    (e_ax_stbd, False),
                ],
                triangles=_grid_triangles(top_grid_pts),
                u_degree=3,
                v_degree=1,
                cp_grid_ids=top_grid_ids,
                u_knots=u_knots_shared,
                u_mults=u_mults_shared,
                v_knots=[0.0, 1.0],
                v_mults=[2, 2],
            )
        )

        # Bottom shell (u: port -> stbd via -z, v: sec i -> sec i+1)
        bot_grid_ids = [[a, b] for a, b in zip(bot_cp_ids_sec[i], bot_cp_ids_sec[i + 1])]
        bot_grid_pts = np.stack([bot_samples_sec[i], bot_samples_sec[i + 1]], axis=1)
        faces.append(
            _SolidFace(
                kind="bspline",
                loop=[
                    (e_bot_idx[i], True),
                    (e_ax_stbd, True),
                    (e_bot_idx[i + 1], False),
                    (e_ax_port, False),
                ],
                triangles=_grid_triangles(bot_grid_pts),
                u_degree=3,
                v_degree=1,
                cp_grid_ids=bot_grid_ids,
                u_knots=u_knots_shared,
                u_mults=u_mults_shared,
                v_knots=[0.0, 1.0],
                v_mults=[2, 2],
            )
        )

    # Planar nose (i=0) and tail (i=n_xsecs-1) endcaps
    for sec_idx in (0, n_xsecs - 1):
        poly_pts = np.concatenate(
            [top_samples_sec[sec_idx][:-1], bot_samples_sec[sec_idx][:-1]], axis=0
        )
        n_poly, cap_tris = _polygon_normal_and_triangles(centers_mm[sec_idx], poly_pts)
        origin_id = writer.cartesian_point(centers_mm[sec_idx])
        faces.append(
            _SolidFace(
                kind="plane",
                loop=[(e_top_idx[sec_idx], True), (e_bot_idx[sec_idx], True)],
                triangles=cap_tris,
                plane_origin_id=origin_id,
                plane_normal=n_poly,
                plane_ref_dir=np.array([0.0, 1.0, 0.0]),
            )
        )

    return writer.emit_solid(solid_name, vertices_xyz, edges, faces)


def _unfold_centerline_symmetric_wing(wing: Any) -> Wing:
    """Unfold a centerline-rooted symmetric wing into a single port-to-starboard Wing."""
    port_xsecs = []
    for xs in reversed(wing.xsecs[1:]):
        xs_m = copy.deepcopy(xs)
        xs_m.xyz_le = np.array(
            [float(xs.xyz_le[0]), -float(xs.xyz_le[1]), float(xs.xyz_le[2])],
            dtype=float,
        )
        port_xsecs.append(xs_m)

    root_xs = copy.deepcopy(wing.xsecs[0])
    root_xs.xyz_le = np.array(
        [float(root_xs.xyz_le[0]), 0.0, float(root_xs.xyz_le[2])],
        dtype=float,
    )
    stbd_xsecs = [copy.deepcopy(xs) for xs in wing.xsecs[1:]]
    return Wing(
        name=wing.name,
        symmetric=False,
        xsecs=port_xsecs + [root_xs] + stbd_xsecs,
    )


def export_assembly_step(
    assembly: Any,
    filename: Path | str,
    minimum_airfoil_TE_thickness: float = 0.001,
    fuselage_tol: float = 1e-4,
    n_chordwise: int = 41,
    merge_symmetric_wings: bool = True,
) -> Path:
    """
    Export a ``HydrofoilAssembly`` (or ``Airplane``) to a watertight ISO 10303-21
    STEP AP214 NURBS B-Rep file in millimeters.

    Works with pure NumPy/SciPy (no ``cadquery`` or external CAD dependency needed).
    Each wing and each fuselage is written as an oriented closed
    ``MANIFOLD_SOLID_BREP`` shell with exact shared boundary control points. When
    ``merge_symmetric_wings=True`` (default), centerline-rooted symmetric wings
    are unfolded into a single continuous port-to-starboard solid instead of two
    separate halves split at ``y = 0``.

    Args:
        assembly: A ``HydrofoilAssembly`` or ``Airplane`` object.
        filename: Output ``.step`` / ``.stp`` filepath.
        minimum_airfoil_TE_thickness: Minimum trailing-edge thickness-to-chord
            ratio enforced to prevent zero-thickness CAD edges.
        fuselage_tol: Minimum fuselage cross-section width/height [m].
        n_chordwise: Number of cosine-spaced chordwise stations fitted per airfoil
            skin curve.
        merge_symmetric_wings: Merge centerline-rooted symmetric wings into a
            single continuous full-span ``MANIFOLD_SOLID_BREP`` solid.

    Returns:
        Resolved ``Path`` of the written STEP file.
    """
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prod_name = getattr(assembly, "name", None) or "HydrofoilAssembly"
    writer = _StepWriter(product_name=prod_name)
    solid_ids: list[int] = []
    used_names: set[str] = set()

    def _unique_name(base: str) -> str:
        if base not in used_names:
            used_names.add(base)
            return base
        idx = 2
        while f"{base}_{idx}" in used_names:
            idx += 1
        out = f"{base}_{idx}"
        used_names.add(out)
        return out

    for wing in assembly.wings:
        if len(wing.xsecs) < 2:
            continue
        base_name = wing.name if getattr(wing, "name", None) and wing.name != "Untitled" else "Wing"
        can_merge_centerline = (
            merge_symmetric_wings
            and wing.symmetric
            and abs(float(wing.xsecs[0].xyz_le[1])) <= 1e-6
            and all(float(xs.xyz_le[1]) > 1e-6 for xs in wing.xsecs[1:])
        )
        if can_merge_centerline:
            unfolded = _unfold_centerline_symmetric_wing(wing)
            sid, _ = _add_wing_solid(
                writer=writer,
                wing=unfolded,
                solid_name=_unique_name(base_name),
                mirror_xz=False,
                minimum_airfoil_TE_thickness=minimum_airfoil_TE_thickness,
                n_chordwise=n_chordwise,
            )
            solid_ids.append(sid)
        else:
            sid, _ = _add_wing_solid(
                writer=writer,
                wing=wing,
                solid_name=_unique_name(base_name),
                mirror_xz=False,
                minimum_airfoil_TE_thickness=minimum_airfoil_TE_thickness,
                n_chordwise=n_chordwise,
            )
            solid_ids.append(sid)

            if wing.symmetric:
                sid_m, _ = _add_wing_solid(
                    writer=writer,
                    wing=wing,
                    solid_name=_unique_name(f"{base_name}_mirror"),
                    mirror_xz=True,
                    minimum_airfoil_TE_thickness=minimum_airfoil_TE_thickness,
                    n_chordwise=n_chordwise,
                )
                solid_ids.append(sid_m)

    for fuse in assembly.fuselages:
        if len(fuse.xsecs) < 2:
            continue
        base_name = fuse.name if getattr(fuse, "name", None) and fuse.name != "Untitled" else "Fuselage"
        sid_f, _ = _add_fuselage_solid(
            writer=writer,
            fuselage=fuse,
            solid_name=_unique_name(base_name),
            fuselage_tol=fuselage_tol,
        )
        solid_ids.append(sid_f)

    if not solid_ids:
        raise ValueError("Cannot export empty assembly to STEP (no wings or fuselages with >=2 xsecs).")

    step_text = writer.build_step_text(solid_ids)
    out_path.write_text(step_text, encoding="utf-8")
    return out_path
