"""
Flow fields of a craft (hull + foils) as VTK surface meshes, for ParaView.

``export_craft_vtk`` writes, in earth axes (x aft, y to starboard, z up from the
still water surface, metres):

- ``<prefix>_foils.vtp``: the foils' and struts' surfaces with the section pressure
  and boundary layer at every spanwise station;
- ``<prefix>_hull.vtp``: the hull surfaces with the pressure of the hull's own flow;
- ``<prefix>_surface.vtp``: the free surface, displaced by the wave elevation of
  the hull, foils and pods together;
- ``<prefix>_bodies.vtp``: pods and other bodies of revolution (geometry only);
- ``<prefix>.vtm``: an index that opens them all at once.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as onp

import aerosandbox as asb

_CHORDWISE = 32  # NeuralFoil's boundary-layer stations per surface


def _write_vtp(path: str, points: onp.ndarray, quads: onp.ndarray, fields: dict[str, onp.ndarray]):
    """One PolyData file of quadrilaterals with per-point scalar fields."""
    points = onp.asarray(points, dtype=float).reshape(-1, 3)
    quads = onp.asarray(quads, dtype=int).reshape(-1, 4)

    def text(array, fmt):
        return "\n".join(" ".join(fmt % v for v in row) for row in onp.atleast_2d(array))

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'<PolyData>\n<Piece NumberOfPoints="{len(points)}" NumberOfPolys="{len(quads)}">\n')
        f.write('<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        f.write(text(points, "%.6g") + "\n</DataArray>\n</Points>\n<Polys>\n")
        f.write('<DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write(text(quads, "%d") + "\n</DataArray>\n")
        f.write('<DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write(text((4 * onp.arange(1, len(quads) + 1)).reshape(-1, 1), "%d"))
        f.write("\n</DataArray>\n</Polys>\n<PointData>\n")
        for name, values in fields.items():
            values = onp.asarray(values, dtype=float).reshape(-1)
            if len(values) != len(points):
                raise ValueError(f"Field {name!r} has {len(values)} values for {len(points)} points.")
            f.write(f'<DataArray type="Float32" Name="{name}" format="ascii">\n')
            f.write("\n".join("nan" if not onp.isfinite(v) else "%.6g" % v for v in values))
            f.write("\n</DataArray>\n")
        f.write("</PointData>\n</Piece>\n</PolyData>\n</VTKFile>\n")


def _grid_quads(n_i: int, n_j: int, offset: int = 0) -> onp.ndarray:
    """Quadrilaterals of an ``n_i x n_j`` grid of points numbered row by row."""
    i, j = onp.meshgrid(onp.arange(n_i - 1), onp.arange(n_j - 1), indexing="ij")
    a = (i * n_j + j).reshape(-1) + offset
    return onp.stack([a, a + n_j, a + n_j + 1, a + 1], axis=1)


class _Mesh:
    """Grids of points gathered into one PolyData."""

    def __init__(self):
        self.points, self.quads, self.fields, self.count = [], [], {}, 0

    def add_grid(self, points: onp.ndarray, fields: dict[str, onp.ndarray]):
        n_i, n_j = points.shape[:2]
        self.quads.append(_grid_quads(n_i, n_j, self.count))
        self.points.append(points.reshape(-1, 3))
        for name, values in fields.items():
            self.fields.setdefault(name, []).append(onp.broadcast_to(values, (n_i, n_j)).reshape(-1))
        self.count += n_i * n_j

    def write(self, path: str) -> str:
        _write_vtp(
            path,
            onp.concatenate(self.points),
            onp.concatenate(self.quads),
            {name: onp.concatenate(parts) for name, parts in self.fields.items()},
        )
        return path


def section_boundary_layer(
    airfoil: asb.Airfoil, lift_coefficient: float, reynolds: float, model_size: str = "large"
) -> dict[str, Any]:
    """
    The two-dimensional pressure and boundary layer of a section at the angle of
    attack that gives ``lift_coefficient``, from NeuralFoil.

    Returns ``alpha`` [deg], the achieved ``CL`` and ``CD``, the transition points
    ``xtr_upper``, ``xtr_lower``, and, per surface (``"upper"``, ``"lower"``) at the
    chord fractions ``x``: edge velocity over free-stream ``ue``, momentum thickness
    over chord ``theta``, and shape factor ``H``.
    """
    import neuralfoil as nf

    kulfan = airfoil if isinstance(airfoil, asb.KulfanAirfoil) else airfoil.to_kulfan_airfoil()

    def run(alpha):
        return nf.get_aero_from_kulfan_parameters(
            kulfan.kulfan_parameters, alpha=alpha, Re=reynolds, model_size=model_size
        )

    def cl(alpha):
        return float(onp.ravel(run(alpha)["CL"])[0])

    # Secant iteration from the thin-aerofoil slope, kept within attached flow.
    a0, a1 = 0.0, 2.0
    f0, f1 = cl(a0) - lift_coefficient, cl(a1) - lift_coefficient
    for _ in range(12):
        if abs(f1) < 1e-4 or abs(f1 - f0) < 1e-9:
            break
        a2 = float(onp.clip(a1 - f1 * (a1 - a0) / (f1 - f0), -10.0, 12.0))
        a0, f0, a1 = a1, f1, a2
        f1 = cl(a1) - lift_coefficient
    aero = run(a1)

    def side(name):
        return {
            key: onp.array([float(onp.ravel(aero[f"{name}_bl_{label}_{i}"])[0]) for i in range(_CHORDWISE)])
            for key, label in (("ue", "ue/vinf"), ("theta", "theta"), ("H", "H"))
        }

    return {
        "alpha": a1,
        "CL": float(onp.ravel(aero["CL"])[0]),
        "CD": float(onp.ravel(aero["CD"])[0]),
        "xtr_upper": float(onp.ravel(aero["Top_Xtr"])[0]),
        "xtr_lower": float(onp.ravel(aero["Bot_Xtr"])[0]),
        "x": onp.asarray(nf.bl_x_points, dtype=float),
        "upper": side("upper"),
        "lower": side("lower"),
    }


def _wing_stations(wing: asb.Wing, n_span: int):
    """Spanwise stations of a wing: (xsec index, fraction to the next, arc position)."""
    quarter = [
        onp.array([float(v) for v in wing._compute_xyz_of_WingXSec(i, x_nondim=0.25, z_nondim=0.0)])
        for i in range(len(wing.xsecs))
    ]
    lengths = [float(onp.linalg.norm(quarter[i + 1] - quarter[i])) for i in range(len(quarter) - 1)]
    total = sum(lengths)
    stations, start = [], 0.0
    for i, length in enumerate(lengths):
        n = max(2, int(round(n_span * length / total)))
        # Cosine spacing towards both ends of each panel.
        fractions = 0.5 * (1.0 - onp.cos(onp.linspace(0.0, onp.pi, n + 1)))
        for t in fractions if i == len(lengths) - 1 else fractions[:-1]:
            stations.append((i, float(t), start + t * length))
        start += length
    return stations, total


def _foil_mesh(craft, results, spanwise_loading: dict[str, str], n_span: int, model_size: str) -> _Mesh:
    water, velocity = craft.water, float(craft.velocity)
    rho, g = float(water.density()), float(water.gravity)
    q = 0.5 * rho * velocity**2
    nu = float(water.kinematic_viscosity())
    mesh = _Mesh()
    components = results["foil"]["wing_hydro_components"]
    for wing_id, (wing, component) in enumerate(zip(craft.assembly.wings, components)):
        is_strut = bool(getattr(wing, "is_strut", False))
        lift = 0.0 if is_strut else float(onp.ravel(component.L)[0])
        stations, arc = _wing_stations(wing, n_span)
        area = float(wing.area())
        loading = spanwise_loading.get(wing.name, "elliptic")
        rows = []
        for i, t, s in stations:
            xa, xb = wing.xsecs[i], wing.xsecs[i + 1]
            chord = (1.0 - t) * float(xa.chord) + t * float(xb.chord)
            # Position between the wing's ends: -1 and 1 at free tips.
            eta = s / arc if wing.symmetric else 2.0 * s / arc - 1.0
            span = 2.0 * arc if wing.symmetric else arc
            if is_strut or loading == "uniform":
                cl = lift / (q * area)
            else:
                cl = 4.0 * lift / (onp.pi * span) * onp.sqrt(max(1.0 - eta**2, 0.0)) / (q * chord)
            cl = float(onp.clip(cl, -1.2, 1.2))
            airfoil = xa.airfoil if t < 0.5 else xb.airfoil
            bl = section_boundary_layer(airfoil, cl, velocity * chord / nu, model_size)
            kulfan = airfoil if isinstance(airfoil, asb.KulfanAirfoil) else airfoil.to_kulfan_airfoil()

            # Around the section: upper trailing edge -> leading edge -> lower trailing edge.
            x = bl["x"]
            xc = onp.concatenate([[1.0], x[::-1], [0.0], x, [1.0]])
            camber = onp.asarray(kulfan.local_camber(xc), dtype=float).reshape(-1)
            thickness = onp.asarray(kulfan.local_thickness(xc), dtype=float).reshape(-1)
            upper = onp.arange(len(xc)) <= _CHORDWISE + 1
            zc = camber + onp.where(upper, 0.5, -0.5) * thickness

            def around(key, at_nose, at_tail=None):
                u, low = bl["upper"][key], bl["lower"][key]
                return onp.concatenate([
                    [u[-1] if at_tail is None else at_tail], u[::-1], [at_nose], low,
                    [low[-1] if at_tail is None else at_tail],
                ])

            ue = onp.abs(around("ue", 0.0))  # (NeuralFoil signs the lower surface's)
            theta = around("theta", 0.5 * (bl["upper"]["theta"][0] + bl["lower"]["theta"][0])) * chord
            shape = around("H", 0.5 * (bl["upper"]["H"][0] + bl["lower"]["H"][0]))
            cp = 1.0 - ue**2

            points_g = onp.array([
                [
                    (1.0 - t) * float(pa) + t * float(pb)
                    for pa, pb in zip(
                        wing._compute_xyz_of_WingXSec(i, x_nondim=float(xn), z_nondim=float(zn)),
                        wing._compute_xyz_of_WingXSec(i + 1, x_nondim=float(xn), z_nondim=float(zn)),
                    )
                ]
                for xn, zn in zip(xc, zc)
            ])
            # Streamwise pressure gradient along the surface, from the nose on each side.
            arc_len = onp.concatenate([[0.0], onp.cumsum(onp.linalg.norm(onp.diff(points_g, axis=0), axis=1))])
            nose = _CHORDWISE + 1
            gradient = onp.gradient(q * cp, arc_len) * onp.where(onp.arange(len(xc)) < nose, -1.0, 1.0)
            gradient[nose] = 0.0
            xtr = onp.where(upper, bl["xtr_upper"], bl["xtr_lower"])
            rows.append({
                "points": points_g,
                "pressure_coefficient": cp,
                "pressure_dynamic": q * cp,
                "pressure_gradient": gradient,
                "edge_velocity_ratio": ue,
                "momentum_thickness": theta,
                "shape_factor": shape,
                "displacement_thickness": shape * theta,
                "turbulent": (xc > xtr).astype(float),
                "chord_fraction": xc,
                "upper_side": upper.astype(float),
                "section_lift_coefficient": onp.full(len(xc), bl["CL"]),
                "section_angle_of_attack": onp.full(len(xc), bl["alpha"]),
                "section_reynolds": onp.full(len(xc), velocity * chord / nu),
            })
        points = onp.stack([row["points"] for row in rows])  # (span, around, 3)
        fields = {name: onp.stack([row[name] for row in rows]) for name in rows[0] if name != "points"}
        fields["component"] = onp.full(points.shape[:2], float(wing_id))
        for mirror in ((1.0, -1.0) if wing.symmetric else (1.0,)):
            x_e, z_e = craft._earth(points[..., 0], points[..., 2])
            earth = onp.stack([onp.asarray(x_e, dtype=float), mirror * points[..., 1], onp.asarray(z_e, dtype=float)], axis=-1)
            depth = -earth[..., 2]
            mesh.add_grid(earth, dict(
                fields,
                depth=depth,
                pressure_gauge=rho * g * depth + fields["pressure_dynamic"],
            ))
    return mesh


def _bodies_mesh(craft, n_around: int = 24) -> _Mesh:
    mesh = _Mesh()
    angles = onp.linspace(0.0, 2.0 * onp.pi, n_around + 1)
    for body in craft.assembly.fuselages:
        rings = []
        for xsec in body.xsecs:
            centre = [float(v) for v in xsec.xyz_c]
            radius = float(xsec.equivalent_radius())
            x_e, z_e = craft._earth(
                onp.full_like(angles, centre[0]), centre[2] + radius * onp.sin(angles)
            )
            rings.append(onp.stack([onp.asarray(x_e, dtype=float), centre[1] + radius * onp.cos(angles),
                                    onp.asarray(z_e, dtype=float)], axis=-1))
        points = onp.stack(rings)
        mesh.add_grid(points, {"depth": -points[..., 2]})
    return mesh


def export_craft_vtk(
    prefix: str,
    craft,
    spanwise_loading: dict[str, str] | None = None,
    n_span: int = 24,
    surface_extent: Sequence[float] = (0.5, 2.5, 1.2),
    surface_spacing: float | None = None,
    wave_level: int = 1,
    k_cap: float = 100.0,
    surface_wave_level: int = 2,
    surface_k_cap: float = 12.0,
    model_size: str = "large",
) -> dict[str, str]:
    """
    Write the flow fields of a ``CraftBuildup`` at its (numeric) attitude as VTK
    surface meshes; see this module's description. Returns the files written.

    Parameters
    ----------
    prefix : Path and file-name stem; ``_foils.vtp`` etc. are appended.
    craft : A ``hydrosandbox.hull.CraftBuildup`` with numeric inputs, normally at
        its equilibrium sinkage and trim.
    spanwise_loading : Per wing name, ``"elliptic"`` (default: the wing's lift
        spread elliptically between its tips) or ``"uniform"`` (the same section
        lift coefficient everywhere, e.g. a foil held between two struts).
    n_span : Spanwise stations per wing.
    surface_extent : The free-surface patch, in hull lengths: ahead of the bow,
        astern of the stern, and to each side of the centreline.
    surface_spacing : Its grid spacing [m]; a sixth of the shortest wave kept
        (``2 pi / surface_k_cap``) if None.
    wave_level, k_cap : Quadrature refinement and largest wavenumber [1/m] of the
        wave fields at the hull (see ``Hull.wave_field``).
    surface_wave_level, surface_k_cap : The same for the free-surface patch. Far
        from the craft the quadrature only resolves longer waves, so the cap is
        lower here: waves shorter than ``2 pi / surface_k_cap`` are smoothed away.
        (At the defaults the spurious elevation ahead of a 5 m craft is under a
        millimetre; at ``k_cap = 100`` it is centimetres.)

    What the fields are
    -------------------
    Foils: at each spanwise station, the two-dimensional section solution
    (NeuralFoil) at the section lift coefficient implied by the wing's lift and the
    chosen spanwise loading. So the boundary layer is strip-wise: no crossflow, and
    nothing of the junctions. ``pressure_gradient`` [Pa/m] is along the surface in
    the flow direction, positive adverse. Thicknesses are in metres.

    Hull: linearised pressure ``-rho U u`` from the hull's own slender-body flow and
    its wave field, and the wave field of the foils and pods. It is set to NaN above
    the local water surface. The hull has no boundary-layer solution.

    Free surface: linear wave elevation [m] of the hull (``elevation_hull``) and of
    the foils, struts and pods (``elevation_appendages``), and their sum, by which
    the surface is displaced.
    """
    from hydrosandbox.hull.hull import wave_field_from_source_amplitude
    from hydrosandbox.library.wave_resistance import element_source_amplitude

    hull, water = craft.hull, craft.water
    velocity = float(craft.velocity)
    rho, g = float(water.density()), float(water.gravity)
    sinkage, trim = float(craft.sinkage), float(craft.trim)
    pivot = (float(craft.xyz_cg[0]), float(craft.xyz_cg[2]))
    spacing = craft.demihull_spacing
    results = craft.run()
    written: dict[str, str] = {}
    directory = os.path.dirname(os.path.abspath(prefix))
    os.makedirs(directory, exist_ok=True)

    ### Wave fields: the hull's, and the foils', struts' and pods' together.
    amplitude = None
    elements = None if craft.assembly is None else results["foil"]["wave_elements"]
    if elements is not None and len(elements["source_flux"]) > 0:
        def earth(points):
            x_e, z_e = craft._earth(onp.array([float(v) for v in points[0]]), onp.array([float(v) for v in points[2]]))
            return onp.asarray(x_e, dtype=float), onp.array([float(v) for v in points[1]]), onp.asarray(z_e, dtype=float)

        numbers = {key: onp.array([float(onp.ravel(v)[0]) for v in elements[key]])
                   for key in ("source_flux", "force_y", "force_z", "smear_length")}
        # (Element forces are in geometry axes; at a degree or two of trim the
        # difference from earth axes is neglected here.)
        amplitude = element_source_amplitude(earth(elements["xyz_a"]), earth(elements["xyz_b"]), velocity, rho, **numbers)

    def wave_fields(cap, level):
        """(hull's, appendages' or None), for wavenumbers up to ``cap``."""
        both = [
            hull.wave_field(velocity, g, sinkage, trim, pivot, demihull_spacing=spacing, k_cap=cap, level=level),
            None if amplitude is None else wave_field_from_source_amplitude(
                amplitude, velocity, g, length=hull.length, k_cap=cap, level=level
            ),
        ]
        for field in both:
            if field is not None:
                field.chunk = 256
        return both

    fields = [f for f in wave_fields(k_cap, wave_level) if f is not None]

    def elevation(field, x, y):
        return onp.asarray(field.elevation(onp.ravel(x), onp.ravel(y)), dtype=float).reshape(onp.shape(x))

    ### Foils and struts.
    if craft.assembly is not None and craft.assembly.wings:
        written["foils"] = _foil_mesh(craft, results, dict(spanwise_loading or {}), n_span, model_size).write(
            prefix + "_foils.vtp"
        )
    if craft.assembly is not None and craft.assembly.fuselages:
        written["bodies"] = _bodies_mesh(craft).write(prefix + "_bodies.vtp")

    ### Hull surfaces.
    sections = hull.sections(velocity, g, sinkage, trim, pivot)
    x_g, z_g = onp.meshgrid(hull.stations, hull.levels, indexing="ij")
    x_e, z_e = (onp.asarray(v, dtype=float) for v in craft._earth(x_g, z_g))
    mesh = _Mesh()
    for y_hull in ([0.0] if spacing is None else [-0.5 * spacing, 0.5 * spacing]):
        for side in (1.0, -1.0):
            y = y_hull + side * hull.half_breadths
            below = onp.minimum(z_e, 0.0)
            flat = (x_e.reshape(-1), y.reshape(-1), below.reshape(-1))
            u = onp.asarray(hull.induced_velocity(*flat, velocity, g, sinkage, trim, pivot,
                                                  demihull_spacing=spacing, sections=sections)[0], dtype=float).reshape(-1)
            u_wave = sum(onp.asarray(f.induced_velocity(*flat)[0], dtype=float).reshape(-1) for f in fields)
            surface = sum(elevation(f, x_e, y) for f in fields)
            wet = z_e <= surface
            p = (-rho * velocity * (u + u_wave)).reshape(x_e.shape)
            mesh.add_grid(onp.stack([x_e, y, z_e], axis=-1), {
                "pressure_dynamic": onp.where(wet, p, onp.nan),
                "pressure_coefficient": onp.where(wet, p / (0.5 * rho * velocity**2), onp.nan),
                "pressure_gauge": onp.where(wet, p - rho * g * z_e, onp.nan),
                "wetted": wet.astype(float),
                "depth": -z_e,
            })
    written["hull"] = mesh.write(prefix + "_hull.vtp")

    ### Free surface (computed for y >= 0 and mirrored).
    length = hull.length
    step = onp.pi / (3.0 * surface_k_cap) if surface_spacing is None else float(surface_spacing)
    hull_field, appendage_field = wave_fields(surface_k_cap, surface_wave_level)
    x_bow, x_stern = float(hull.stations[0]), float(hull.stations[-1])
    xs = onp.arange(x_bow - surface_extent[0] * length, x_stern + surface_extent[1] * length + 0.5 * step, step)
    ys_half = onp.arange(0.0, surface_extent[2] * length + 0.5 * step, step)
    xx, yy = onp.meshgrid(xs, ys_half, indexing="ij")
    zeta_hull = elevation(hull_field, xx, yy)
    zeta_app = elevation(appendage_field, xx, yy) if appendage_field is not None else onp.zeros_like(xx)

    def mirrored(a, sign=1.0):
        return onp.concatenate([sign * a[:, :0:-1], a], axis=1)

    xx, yy = mirrored(xx), mirrored(yy, -1.0)
    zeta_hull, zeta_app = mirrored(zeta_hull), mirrored(zeta_app)
    mesh = _Mesh()
    mesh.add_grid(onp.stack([xx, yy, zeta_hull + zeta_app], axis=-1), {
        "elevation": zeta_hull + zeta_app,
        "elevation_hull": zeta_hull,
        "elevation_appendages": zeta_app,
    })
    written["surface"] = mesh.write(prefix + "_surface.vtp")

    index = prefix + ".vtm"
    with open(index, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="vtkMultiBlockDataSet" version="1.0">\n<vtkMultiBlockDataSet>\n')
        for n, (name, path) in enumerate(written.items()):
            f.write(f'<DataSet index="{n}" name="{name}" file="{os.path.basename(path)}"/>\n')
        f.write("</vtkMultiBlockDataSet>\n</VTKFile>\n")
    written["index"] = index
    return written
