import copy
from typing import Any, Literal, Sequence
from aerosandbox.geometry.airfoil import Airfoil
from aerosandbox.geometry.fuselage import Fuselage
from aerosandbox.geometry.wing import Wing, WingXSec
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.library.free_surface import smooth_submergence_fraction


def wing_submerged_area(
    wing: Wing,
    surface_z: Vectorizable | None = None,
) -> Vectorizable:
    """
    Compute the smoothly submerged planform area [m^2] of any ``Wing`` (``Foil`` or ``Strut``).
    """
    if len(wing.xsecs) < 2:
        return 0.0

    sectional_areas = wing.area(_sectional=True)
    if surface_z is None:
        total = sum(sectional_areas)
        return 2.0 * total if wing.symmetric else total

    total = 0.0
    for i, sect_area in enumerate(sectional_areas):
        z_a = wing.xsecs[i].xyz_le[2]
        z_b = wing.xsecs[i + 1].xyz_le[2]
        f_sub, _, _ = smooth_submergence_fraction(z_a, z_b, surface_z)
        total = total + sect_area * f_sub

    return 2.0 * total if wing.symmetric else total


def wing_submerged_volume_and_centroid(
    wing: Wing,
    surface_z: Vectorizable | None = None,
) -> tuple[Vectorizable, list[Vectorizable]]:
    """
    Compute the smoothly submerged volume [m^3] and centroid ``[x, y, z]`` [m]
    of any ``Wing`` (``Foil`` or ``Strut``).
    """
    if len(wing.xsecs) < 2:
        return 0.0, [0.0, 0.0, 0.0]

    sectional_volumes = wing.volume(_sectional=True)
    xsec_centroids = [
        wing._compute_xyz_of_WingXSec(i, x_nondim=0.40, z_nondim=0.0)
        for i in range(len(wing.xsecs))
    ]

    v_total = 0.0
    moment_total = [0.0, 0.0, 0.0]

    for i, v_sect in enumerate(sectional_volumes):
        c_a = xsec_centroids[i]
        c_b = xsec_centroids[i + 1]
        if surface_z is None:
            f_sub = 1.0
            s_sub = 0.5
        else:
            f_sub, s_sub, _ = smooth_submergence_fraction(
                wing.xsecs[i].xyz_le[2],
                wing.xsecs[i + 1].xyz_le[2],
                surface_z,
            )

        v_sub = v_sect * f_sub
        c_sub = [(1.0 - s_sub) * c_a[k] + s_sub * c_b[k] for k in range(3)]

        if wing.symmetric:
            v_total = v_total + 2.0 * v_sub
            moment_total[0] = moment_total[0] + 2.0 * v_sub * c_sub[0]
            # Symmetric port/starboard y-moments cancel identically.
            moment_total[2] = moment_total[2] + 2.0 * v_sub * c_sub[2]
        else:
            v_total = v_total + v_sub
            for k in range(3):
                moment_total[k] = moment_total[k] + v_sub * c_sub[k]

    v_safe = np.maximum(v_total, 1e-16)
    centroid = [moment_total[k] / v_safe for k in range(3)]
    return v_total, centroid


def fuselage_submerged_volume_and_centroid(
    fuselage: Fuselage,
    surface_z: Vectorizable | None = None,
) -> tuple[Vectorizable, list[Vectorizable]]:
    """
    Compute the smoothly submerged volume [m^3] and centroid ``[x, y, z]`` [m]
    of any ``Fuselage``.
    """
    if len(fuselage.xsecs) < 2:
        return 0.0, [0.0, 0.0, 0.0]

    sectional_volumes = fuselage.volume(_sectional=True)
    v_total = 0.0
    moment_total = [0.0, 0.0, 0.0]

    for i, v_sect in enumerate(sectional_volumes):
        c_a = fuselage.xsecs[i].xyz_c
        c_b = fuselage.xsecs[i + 1].xyz_c
        if surface_z is None:
            f_sub = 1.0
            s_sub = 0.5
        else:
            f_sub, s_sub, _ = smooth_submergence_fraction(
                c_a[2],
                c_b[2],
                surface_z,
            )
        v_sub = v_sect * f_sub
        c_sub = [(1.0 - s_sub) * c_a[k] + s_sub * c_b[k] for k in range(3)]
        v_total = v_total + v_sub
        for k in range(3):
            moment_total[k] = moment_total[k] + v_sub * c_sub[k]

    v_safe = np.maximum(v_total, 1e-16)
    centroid = [moment_total[k] / v_safe for k in range(3)]
    return v_total, centroid


class Foil(Wing):
    """
    Lifting hydrofoil surface (main wing, stabilizer, canard, rudder, etc.).
    """

    def __init__(
        self,
        name: str | None = None,
        xsecs: list[WingXSec] | None = None,
        symmetric: bool = False,
        role: Literal["main", "stabilizer", "canard", "control", "other"] = "main",
        color: (str | tuple[float]) | None = None,
        analysis_specific_options: dict[type, dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            xsecs=xsecs,
            symmetric=symmetric,
            color=color,
            analysis_specific_options=analysis_specific_options,
            **kwargs,
        )
        self.role = role
        self.is_strut = False

    def submerged_area(self, surface_z: Vectorizable | None = None) -> Vectorizable:
        """Return smoothly submerged planform area [m^2]."""
        return wing_submerged_area(self, surface_z=surface_z)

    def submerged_volume_and_centroid(
        self, surface_z: Vectorizable | None = None
    ) -> tuple[Vectorizable, list[Vectorizable]]:
        """Return smoothly submerged volume [m^3] and buoyancy centroid ``[x, y, z]`` [m]."""
        return wing_submerged_volume_and_centroid(self, surface_z=surface_z)


class Strut(Wing):
    """
    Surface-piercing or connecting hydrofoil strut / mast.
    """

    def __init__(
        self,
        name: str | None = None,
        xsecs: list[WingXSec] | None = None,
        symmetric: bool = False,
        color: (str | tuple[float]) | None = None,
        analysis_specific_options: dict[type, dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            xsecs=xsecs,
            symmetric=symmetric,
            color=color,
            analysis_specific_options=analysis_specific_options,
            **kwargs,
        )
        self.is_strut = True

    @classmethod
    def from_dimensions(
        cls,
        name: str = "Strut",
        root_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        length: Vectorizable = 0.85,
        root_chord: Vectorizable = 0.14,
        tip_chord: Vectorizable = 0.12,
        airfoil: Airfoil | None = None,
        cant_angle_deg: Vectorizable = 0.0,
        rake_angle_deg: Vectorizable = 0.0,
        twist_root: Vectorizable = 0.0,
        twist_tip: Vectorizable = 0.0,
        symmetric: bool = False,
    ) -> "Strut":
        """
        Construct a two-section ``Strut`` extending upward (+z_g) from ``root_xyz``.

        Parameters
        ----------
        name : str
            Component name.
        root_xyz : Sequence[float]
            Leading-edge ``[x, y, z]`` coordinates [m] of the bottom (submerged) root.
        length : Vectorizable
            Total strut spanwise length [m].
        root_chord : Vectorizable
            Chord at the bottom root [m].
        tip_chord : Vectorizable
            Chord at the upper tip [m].
        airfoil : Airfoil | None
            Section profile. Defaults to ``Airfoil("naca0012")``.
        cant_angle_deg : Vectorizable
            Outward roll/cant angle [deg] toward +y_g.
        rake_angle_deg : Vectorizable
            Aft rake angle [deg] toward +x_g.
        twist_root : Vectorizable
            Toe/twist angle at the root [deg].
        twist_tip : Vectorizable
            Toe/twist angle at the tip [deg].
        symmetric : bool
            Whether to mirror across the XZ plane.
        """
        if airfoil is None:
            airfoil = Airfoil("naca0012")

        rx, ry, rz = root_xyz[0], root_xyz[1], root_xyz[2]
        dx = length * np.sind(rake_angle_deg)
        dy = length * np.sind(cant_angle_deg) * np.cosd(rake_angle_deg)
        dz = length * np.cosd(cant_angle_deg) * np.cosd(rake_angle_deg)

        tip_xyz = np.array([rx + dx, ry + dy, rz + dz])

        return cls(
            name=name,
            symmetric=symmetric,
            xsecs=[
                WingXSec(
                    xyz_le=np.array([rx, ry, rz]),
                    chord=root_chord,
                    twist=twist_root,
                    airfoil=airfoil,
                ),
                WingXSec(
                    xyz_le=tip_xyz,
                    chord=tip_chord,
                    twist=twist_tip,
                    airfoil=airfoil,
                ),
            ],
        )

    def submerged_area(self, surface_z: Vectorizable | None = None) -> Vectorizable:
        """Return smoothly submerged planform area [m^2]."""
        return wing_submerged_area(self, surface_z=surface_z)

    def submerged_volume_and_centroid(
        self, surface_z: Vectorizable | None = None
    ) -> tuple[Vectorizable, list[Vectorizable]]:
        """Return smoothly submerged volume [m^3] and buoyancy centroid ``[x, y, z]`` [m]."""
        return wing_submerged_volume_and_centroid(self, surface_z=surface_z)


class HydroFuselage(Fuselage):
    """
    Submerged or semi-submerged hydrodynamic pod / fuselage body.
    """

    def submerged_volume_and_centroid(
        self, surface_z: Vectorizable | None = None
    ) -> tuple[Vectorizable, list[Vectorizable]]:
        """Return smoothly submerged volume [m^3] and buoyancy centroid ``[x, y, z]`` [m]."""
        return fuselage_submerged_volume_and_centroid(self, surface_z=surface_z)


def _interpolate_xsec(xsec_a: WingXSec, xsec_b: WingXSec, t: float) -> WingXSec:
    """Linearly interpolate a ``WingXSec`` at fraction ``t`` in ``[0, 1]``."""
    t_f = float(t)
    if t_f <= 0.0:
        return copy.copy(xsec_a)
    if t_f >= 1.0:
        return copy.copy(xsec_b)

    if xsec_a.airfoil is xsec_b.airfoil:
        airfoil = xsec_a.airfoil
    else:
        airfoil = xsec_a.airfoil.blend_with_another_airfoil(
            xsec_b.airfoil,
            blend_fraction=t_f,
        )

    return WingXSec(
        xyz_le=(1.0 - t_f) * np.array(xsec_a.xyz_le) + t_f * np.array(xsec_b.xyz_le),
        chord=(1.0 - t_f) * xsec_a.chord + t_f * xsec_b.chord,
        twist=(1.0 - t_f) * xsec_a.twist + t_f * xsec_b.twist,
        airfoil=airfoil,
        control_surfaces=xsec_a.control_surfaces,
    )


def clip_wing_to_free_surface(
    wing: Wing,
    surface_z: float | None,
    tol: float = 1e-9,
) -> list[Wing]:
    """
    Clip a numeric ``Wing`` / ``Foil`` / ``Strut`` to the submerged half-space
    ``z <= surface_z``.

    Returns a list of contiguous submerged wing objects (empty if entirely above water).
    If ``surface_z is None`` or symbolic (CasADi), returns ``[wing]`` unchanged.
    """
    if surface_z is None or np.is_casadi_type(surface_z):
        return [wing]

    if len(wing.xsecs) < 2:
        return []

    for xsec in wing.xsecs:
        if np.is_casadi_type(xsec.xyz_le, recursive=True):
            return [wing]

    sz = float(surface_z)

    # Fast check if all xsecs are already submerged or all emerged.
    zs = [float(xsec.xyz_le[2]) for xsec in wing.xsecs]
    if all(z <= sz + tol for z in zs):
        return [wing]
    if all(z >= sz - tol for z in zs):
        return []

    chains: list[list[WingXSec]] = []
    current_chain: list[WingXSec] = (
        [copy.copy(wing.xsecs[0])] if zs[0] <= sz + tol else []
    )

    for xsec_a, xsec_b, z_a, z_b in zip(
        wing.xsecs[:-1], wing.xsecs[1:], zs[:-1], zs[1:]
    ):
        sub_a = z_a <= sz + tol
        sub_b = z_b <= sz + tol

        if sub_a and sub_b:
            current_chain.append(copy.copy(xsec_b))
        elif sub_a and not sub_b:
            dz = z_b - z_a
            t = float(np.clip((sz - z_a) / dz, 0.0, 1.0)) if abs(dz) > tol else 0.0
            if t > 1e-7:
                current_chain.append(_interpolate_xsec(xsec_a, xsec_b, t))
            if len(current_chain) >= 2:
                chains.append(current_chain)
            current_chain = []
        elif not sub_a and sub_b:
            dz = z_b - z_a
            t = float(np.clip((sz - z_a) / dz, 0.0, 1.0)) if abs(dz) > tol else 0.0
            xsec_wl = _interpolate_xsec(xsec_a, xsec_b, t)
            current_chain = [xsec_wl]
            if 1.0 - t > 1e-7:
                current_chain.append(copy.copy(xsec_b))

    if len(current_chain) >= 2:
        chains.append(current_chain)

    clipped_wings: list[Wing] = []
    for chain in chains:
        deduped = [chain[0]]
        for xsec in chain[1:]:
            if np.linalg.norm(np.array(xsec.xyz_le) - np.array(deduped[-1].xyz_le)) > 1e-8:
                deduped.append(xsec)

        if len(deduped) < 2:
            continue

        # Check non-zero YZ span so Wing._compute_frame_of_WingXSec stays well-defined.
        has_nonzero_span = any(
            (float(b.xyz_le[1]) - float(a.xyz_le[1])) ** 2
            + (float(b.xyz_le[2]) - float(a.xyz_le[2])) ** 2
            > 1e-14
            for a, b in zip(deduped[:-1], deduped[1:])
        )
        if not has_nonzero_span:
            continue

        new_wing = copy.copy(wing)
        new_wing.xsecs = deduped
        clipped_wings.append(new_wing)

    return clipped_wings
