import copy
from pathlib import Path
from typing import Any, Literal, Sequence
from aerosandbox.geometry.airplane import Airplane
from aerosandbox.geometry.fuselage import Fuselage
from aerosandbox.geometry.wing import Wing
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable
from hydrosandbox.fluid import Water
from hydrosandbox.geometry.components import (
    clip_wing_to_free_surface,
    fuselage_submerged_volume_and_centroid,
    wing_submerged_volume_and_centroid,
)
from hydrosandbox.geometry.step_export import export_assembly_step


class HydrofoilAssembly(Airplane):
    """
    Definition for a lifting hydrofoil assembly (foils, struts, and optional pods/fuselages).

    Subclasses ``asb.Airplane`` so all AeroSandbox geometry, meshing, and visualization
    utilities remain compatible, while distinguishing lifting ``foils`` from surface-piercing
    or connecting ``struts`` and providing hydrostatic volume, center of buoyancy, and
    free-surface clipping.
    """

    def __init__(
        self,
        name: str | None = None,
        xyz_ref: np.ndarray | Sequence[float] | None = None,
        foils: Sequence[Wing] | None = None,
        struts: Sequence[Wing] | None = None,
        fuselages: Sequence[Fuselage] | None = None,
        wings: Sequence[Wing] | None = None,
        s_ref: float | None = None,
        c_ref: float | None = None,
        b_ref: float | None = None,
        analysis_specific_options: dict[type, dict[str, Any]] | None = None,
    ):
        """
        Define a new HydrofoilAssembly.

        Parameters
        ----------
        name : str | None
            Assembly name.
        xyz_ref : np.ndarray | Sequence[float] | None
            Moment reference point ``[x, y, z]`` in geometry axes [m].
        foils : Sequence[Wing] | None
            Lifting hydrofoils (main foil, stabilizer, canard, etc.).
        struts : Sequence[Wing] | None
            Vertical or canted surface-piercing / connecting struts.
        fuselages : Sequence[Fuselage] | None
            Submerged pods or fuselages.
        wings : Sequence[Wing] | None
            Optional combined wing sequence (merged with ``foils`` if provided).
        s_ref, c_ref, b_ref : float | None
            Reference area [m^2], chord [m], and span [m]. Defaults to those of
            the first lifting foil (falling back to struts or fuselages if no foils exist).
        analysis_specific_options : dict[type, dict[str, Any]] | None
            Optional per-analysis options passed through to ``Airplane``.
        """
        foil_list: list[Wing] = []
        for w in list(foils or []) + list(wings or []):
            if getattr(w, "is_strut", False):
                continue
            foil_list.append(w)

        strut_list: list[Wing] = []
        for s in list(struts or []):
            if not getattr(s, "is_strut", False):
                s = copy.copy(s)
                s.is_strut = True
            strut_list.append(s)

        for w in list(wings or []):
            if getattr(w, "is_strut", False):
                strut_list.append(w)

        all_wings = foil_list + strut_list

        # Default reference quantities come from the primary lifting foil rather than a strut.
        if foil_list:
            primary = foil_list[0]
            if s_ref is None:
                s_ref = primary.area()
            if c_ref is None:
                c_ref = primary.mean_aerodynamic_chord()
            if b_ref is None:
                b_ref = primary.span(include_centerline_distance=True)

        super().__init__(
            name=name,
            xyz_ref=xyz_ref,
            wings=all_wings,
            fuselages=list(fuselages or []),
            s_ref=s_ref,
            c_ref=c_ref,
            b_ref=b_ref,
            analysis_specific_options=analysis_specific_options,
        )

    def __repr__(self) -> str:
        n_foils = len(self.foils)
        n_struts = len(self.struts)
        n_fuses = len(self.fuselages)
        return (
            f"HydrofoilAssembly '{self.name}' "
            f"({n_foils} {'foil' if n_foils == 1 else 'foils'}, "
            f"{n_struts} {'strut' if n_struts == 1 else 'struts'}, "
            f"{n_fuses} {'fuselage' if n_fuses == 1 else 'fuselages'})"
        )

    @property
    def foils(self) -> list[Wing]:
        """Return all lifting foils in this assembly."""
        return [w for w in self.wings if not getattr(w, "is_strut", False)]

    @property
    def struts(self) -> list[Wing]:
        """Return all struts/masts in this assembly."""
        return [w for w in self.wings if getattr(w, "is_strut", False)]

    def submerged_volume_and_centroid(
        self,
        surface_z: Vectorizable | None = None,
    ) -> tuple[Vectorizable, list[Vectorizable]]:
        """
        Compute total smoothly submerged displaced volume [m^3] and center of
        buoyancy ``[x, y, z]`` [m] across all foils, struts, and fuselages.
        """
        v_total = 0.0
        moment_total = [0.0, 0.0, 0.0]

        for wing in self.wings:
            v_w, c_w = wing_submerged_volume_and_centroid(wing, surface_z=surface_z)
            v_total = v_total + v_w
            for k in range(3):
                moment_total[k] = moment_total[k] + v_w * c_w[k]

        for fuse in self.fuselages:
            v_f, c_f = fuselage_submerged_volume_and_centroid(fuse, surface_z=surface_z)
            v_total = v_total + v_f
            for k in range(3):
                moment_total[k] = moment_total[k] + v_f * c_f[k]

        v_safe = np.maximum(v_total, 1e-16)
        centroid = [moment_total[k] / v_safe for k in range(3)]
        return v_total, centroid

    def displaced_volume(
        self,
        surface_z: Vectorizable | None = None,
    ) -> Vectorizable:
        """Return the total submerged displaced volume [m^3]."""
        v_total, _ = self.submerged_volume_and_centroid(surface_z=surface_z)
        return v_total

    def center_of_buoyancy(
        self,
        surface_z: Vectorizable | None = None,
    ) -> list[Vectorizable]:
        """Return the submerged center of buoyancy ``[x, y, z]`` [m] in geometry axes."""
        _, centroid = self.submerged_volume_and_centroid(surface_z=surface_z)
        return centroid

    def hydrostatic_forces(
        self,
        water: Water,
        surface_z: Vectorizable | None = None,
        xyz_ref: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """
        Compute Archimedes hydrostatic buoyancy force and moment about ``xyz_ref``.

        Buoyancy acts vertically upward (``+z_g`` in geometry axes).
        """
        v_disp, xyz_cb = self.submerged_volume_and_centroid(surface_z=surface_z)
        buoyancy = water.density() * water.gravity * v_disp
        f_buoyancy_g = [0.0, 0.0, buoyancy]

        ref = self.xyz_ref if xyz_ref is None else xyz_ref
        r = [xyz_cb[k] - ref[k] for k in range(3)]
        m_buoyancy_g = list(np.cross(r, f_buoyancy_g, manual=True))

        return {
            "V_displaced": v_disp,
            "xyz_cb": xyz_cb,
            "buoyancy_force": buoyancy,
            "F_buoyancy_g": f_buoyancy_g,
            "M_buoyancy_g": m_buoyancy_g,
        }

    def clipped_to_free_surface(
        self,
        surface_z: float | None,
    ) -> "HydrofoilAssembly":
        """
        Return a shallow copy of this assembly whose wings/struts are clipped to
        ``z <= surface_z``, preserving the parent assembly's ``s_ref``, ``c_ref``,
        ``b_ref``, and ``xyz_ref``.
        """
        if surface_z is None or np.is_casadi_type(surface_z):
            return self

        clipped = copy.copy(self)
        clipped_wings: list[Wing] = []
        for wing in self.wings:
            clipped_wings.extend(clip_wing_to_free_surface(wing, surface_z=surface_z))
        clipped.wings = clipped_wings
        return clipped

    def export_step(
        self,
        filename: Path | str,
        backend: Literal["native", "cadquery"] = "native",
        minimum_airfoil_TE_thickness: float = 0.001,
        fuselage_tol: float = 1e-4,
        n_chordwise: int = 41,
        merge_symmetric_wings: bool = True,
    ) -> Path:
        """
        Export this ``HydrofoilAssembly`` to an ISO 10303-21 STEP file (units: mm).

        Args:
            filename: Output ``.step`` / ``.stp`` filepath.
            backend: ``"native"`` (default, uses HydroSandbox's built-in NURBS
                B-Rep writer with no external CAD dependency) or ``"cadquery"``
                (delegates to ``Airplane.export_cadquery_geometry``).
            minimum_airfoil_TE_thickness: Minimum trailing-edge thickness-to-chord
                ratio enforced on each wing section.
            fuselage_tol: Minimum cross-section width/height [m] for fuselages.
            n_chordwise: Chordwise spline resolution per airfoil surface (native backend).
            merge_symmetric_wings: When ``True`` (default, native backend),
                unfolds centerline-rooted symmetric wings into a single
                continuous port-to-starboard ``MANIFOLD_SOLID_BREP`` solid.

        Returns:
            Resolved ``Path`` of the saved STEP file.
        """
        if backend == "cadquery":
            self.export_cadquery_geometry(
                filename=filename,
                minimum_airfoil_TE_thickness=minimum_airfoil_TE_thickness,
            )
            return Path(filename)
        if backend == "native":
            return export_assembly_step(
                assembly=self,
                filename=filename,
                minimum_airfoil_TE_thickness=minimum_airfoil_TE_thickness,
                fuselage_tol=fuselage_tol,
                n_chordwise=n_chordwise,
                merge_symmetric_wings=merge_symmetric_wings,
            )
        raise ValueError(
            f"Bad STEP export backend {backend!r}. Expected 'native' or 'cadquery'."
        )
