"""
Electric motor performance models, marine mass regressions, and catalog matcher.

Combines:
1. Smooth, CasADi-differentiable circuit and 4-term polynomial loss models
   (``motor_performance_drela``, ``motor_performance_loss``,
   ``CatalogMotor.performance_smooth``, ``CatalogMotor.envelope_margins``) for
   simultaneous optimization with ``aerosandbox.Opti``.
2. Marine mass scaling correlations fitted across axial-flux, radial-flux,
   direct-drive torque, and waterproof pod outrunners (``mass_motor_marine``,
   ``mass_controller_marine``, ``mass_gearbox_marine``, ``mass_battery_marine``).
3. Commercial motor catalog (159 records vendored from ``boatmath`` / ``propopt``,
   plus optional small waterproof foil-assist pod motors) with continuous/peak
   envelope checking, gearbox reduction search, and curve matching.
"""

from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Sequence
import numpy as _onp
import aerosandbox.numpy as np
from aerosandbox.numpy.typing import Vectorizable

GEAR_ETA_DEFAULT: float = 0.97
CONTROLLER_ETA_DEFAULT: float = 0.97
SLACK: float = 1.0001
_KB_HOT_FACTOR: float = 0.01654  # ~= 3 * sqrt(3) / (100 * pi) [N*m/A per V_ll/krpm]

_DEFAULT_DB_PATH = Path(__file__).parent / "data" / "motors.json.gz"


# ---------------------------------------------------------------------------
# Differentiable circuit, loss, and mass models (asb.numpy / Opti compatible)
# ---------------------------------------------------------------------------


def motor_performance_drela(
    voltage: Vectorizable | None = None,
    current: Vectorizable | None = None,
    rpm: Vectorizable | None = None,
    torque: Vectorizable | None = None,
    kv: float = 150.0,
    resistance: float = 0.1,
    no_load_current: float = 1.0,
) -> dict[str, Vectorizable]:
    """
    Drela's first-order 3-parameter DC/BLDC motor model (``Kv``, ``R``, ``I0``).

    Provide EXACTLY TWO of ``voltage``, ``current``, ``rpm``, ``torque``
    (excluding the ill-posed ``(current, torque)`` pair).

    Unlike AeroSandbox's ``fabs(electrical_power - shaft_power)``, waste heat
    is computed directly as ``electrical_power - shaft_power`` so derivatives
    stay smooth everywhere.
    """
    voltage_known = voltage is not None
    current_known = current is not None
    rpm_known = rpm is not None
    torque_known = torque is not None

    if (voltage_known + current_known + rpm_known + torque_known) != 2:
        raise ValueError("Provide exactly two of: voltage, current, rpm, torque.")
    if current_known and torque_known:
        raise ValueError(
            "Cannot supply the combination of current and torque - ill-posed problem."
        )

    kv_rad = kv * math.pi / 30.0
    kt = 1.0 / kv_rad

    while not (voltage_known and current_known and rpm_known and torque_known):
        if rpm_known and current_known and not voltage_known:
            speed = rpm * math.pi / 30.0
            voltage = speed / kv_rad + current * resistance
            voltage_known = True

        if torque_known and not current_known:
            current = torque * kv_rad + no_load_current
            current_known = True

        if voltage_known:
            if rpm_known and not current_known:
                back_emf = (rpm * math.pi / 30.0) / kv_rad
                current = (voltage - back_emf) / resistance
                current_known = True
            if current_known and not rpm_known:
                back_emf = voltage - current * resistance
                rpm = back_emf * kv_rad * 30.0 / math.pi
                rpm_known = True

        if current_known and not torque_known:
            torque = (current - no_load_current) / kv_rad
            torque_known = True

    omega = rpm * math.pi / 30.0
    shaft_power = omega * torque
    electrical_power = voltage * current
    waste_heat = electrical_power - shaft_power
    efficiency = shaft_power / np.fmax(electrical_power, 1e-12)

    return {
        "voltage": voltage,
        "current": current,
        "rpm": rpm,
        "omega": omega,
        "torque": torque,
        "kt": kt,
        "shaft_power": shaft_power,
        "P_shaft": shaft_power,
        "electrical_power": electrical_power,
        "P_elec": electrical_power,
        "efficiency": efficiency,
        "eta_motor": efficiency,
        "waste_heat": waste_heat,
    }


def motor_performance_loss(
    rpm: Vectorizable,
    torque: Vectorizable,
    c_cu: float,
    k_h: float = 0.0,
    k_e: float = 0.0,
    k_0: float = 0.0,
    kt: float | None = None,
    kv: float | None = None,
    r_ll: float = 0.0,
    i0: float = 0.0,
    v_bus: Vectorizable | None = None,
    controller_efficiency: float = CONTROLLER_ETA_DEFAULT,
    eff_includes_controller: bool = False,
    topology: str = "pmsm",
    stack: int = 1,
) -> dict[str, Vectorizable]:
    """
    Smooth 4-term motor loss model following ``boatmath`` / ``propopt``:

        ``P_loss = n_stack * (c_cu * (Q/n)^2 + k_h * omega + k_e * omega^2 + k_0)``

    Fully compatible with ``aerosandbox.numpy`` and ``asb.Opti``.
    """
    n_stack = max(float(stack), 1.0)
    omega = 2.0 * math.pi * rpm / 60.0
    qm = torque / n_stack
    p_shaft = torque * omega

    p_loss_raw = n_stack * (c_cu * qm**2 + k_h * omega + k_e * omega**2 + k_0)
    p_loss_motor = np.softmax(p_loss_raw, 1e-6, softness=1e-4)
    p_elec_motor = p_shaft + p_loss_motor
    eta_motor = p_shaft / np.fmax(p_elec_motor, 1e-9)

    ce = 1.0 if eff_includes_controller else controller_efficiency
    eta_drive = eta_motor * ce
    p_elec = p_elec_motor / np.fmax(ce, 1e-9)
    p_loss_controller = p_elec - p_elec_motor

    i_arms: Vectorizable | None = (qm / kt + i0) if (kt is not None and kt != 0.0) else None
    v_req: Vectorizable | None = None
    if topology.lower() == "pmsm" and kt is not None and kt != 0.0:
        kb_hot = kt / _KB_HOT_FACTOR
        v_ll = kb_hot * rpm / 1000.0 + (i_arms if i_arms is not None else 0.0) * r_ll
        v_req = math.sqrt(2.0) * v_ll
    elif kv is not None and kv != 0.0:
        v_req = rpm / kv + (i_arms if i_arms is not None else 0.0) * r_ll

    i_dc_min_bus = (p_elec / np.fmax(v_req, 1e-6)) if v_req is not None else None
    i_dc_pack = (p_elec / np.fmax(v_bus, 1e-6)) if v_bus is not None else i_dc_min_bus

    return {
        "rpm": rpm,
        "omega": omega,
        "torque": torque,
        "P_shaft": p_shaft,
        "shaft_power": p_shaft,
        "P_loss_motor": p_loss_motor,
        "P_loss_controller": p_loss_controller,
        "P_elec_motor": p_elec_motor,
        "P_elec": p_elec,
        "electrical_power": p_elec,
        "eta_motor": eta_motor,
        "efficiency": eta_motor,
        "eta_drive": eta_drive,
        "I_arms": i_arms,
        "V_req": v_req,
        "I_dc_pack": i_dc_pack,
        "I_dc_min_bus": i_dc_min_bus,
    }


def mass_motor_marine(
    max_cont_torque_nm: Vectorizable,
    topology: Literal[
        "axial_pmsm",
        "direct_drive_torque",
        "radial_pmsm",
        "outrunner_pod",
    ] = "axial_pmsm",
) -> Vectorizable:
    """
    Estimate marine electric motor mass [kg] from continuous rated torque [N*m].

    Power-law fits calibrated against the vendored motor database (motor mass
    scales primarily with continuous torque rather than shaft power).
    """
    q = np.softmax(max_cont_torque_nm, 1e-4, softness=1e-3)
    if topology == "axial_pmsm":
        return 0.3420 * q**0.7740
    elif topology == "direct_drive_torque":
        return 1.5548 * q**0.7754
    elif topology == "radial_pmsm":
        return 0.680 * q**0.775
    elif topology == "outrunner_pod":
        return 0.450 * q**0.775 + 0.15
    else:
        raise ValueError(f"Unknown marine motor topology: {topology!r}")


def mass_controller_marine(
    max_power_w: Vectorizable,
    waterproof: bool = True,
) -> Vectorizable:
    """Estimate marine motor controller / ESC mass [kg] from continuous/peak power [W]."""
    p = np.softmax(max_power_w, 0.0, softness=1.0)
    if waterproof:
        return 3.5e-5 * p + 0.12
    return 2.38e-5 * p


def mass_gearbox_marine(
    max_shaft_power_w: Vectorizable,
    rpm_in: Vectorizable,
    rpm_out: Vectorizable,
) -> Vectorizable:
    """
    Estimate sealed planetary marine reduction gearbox mass [kg]. Returns 0
    for direct drive (numeric ``rpm_in == rpm_out``).
    """
    is_sym = any(
        bool(np.is_casadi_type(v, recursive=True))
        for v in (max_shaft_power_w, rpm_in, rpm_out)
    )
    if not is_sym and abs(float(rpm_in) - float(rpm_out)) < 1e-9:
        return 0.0
    p_kw = np.softmax(max_shaft_power_w / 1000.0, 0.0, softness=1e-3)
    rpm_out_safe = np.fmax(rpm_out, 10.0)
    ratio = np.fmax(rpm_in / rpm_out_safe, 1.0)
    return 0.042 * (p_kw / np.fmax(rpm_out_safe / 1000.0, 0.01)) ** 0.75 * ratio**0.15


def mass_battery_marine(
    capacity_wh: Vectorizable,
    chemistry: Literal["lfp", "nmc"] = "lfp",
    cell_specific_energy_wh_kg: float | None = None,
    pack_cell_fraction: float | None = None,
) -> Vectorizable:
    """
    Estimate marine IP67 battery pack mass [kg] from energy capacity [W*h].
    """
    defaults = {
        "lfp": (165.0, 0.70),
        "nmc": (250.0, 0.68),
    }
    if chemistry not in defaults:
        raise ValueError(f"Unknown battery chemistry {chemistry!r}; choose 'lfp' or 'nmc'.")
    se_default, frac_default = defaults[chemistry]
    se = se_default if cell_specific_energy_wh_kg is None else float(cell_specific_energy_wh_kg)
    frac = frac_default if pack_cell_fraction is None else float(pack_cell_fraction)
    return capacity_wh / (se * frac)


# ---------------------------------------------------------------------------
# Catalog database & discrete matching (boatmath::propeller::motor)
# ---------------------------------------------------------------------------


_FOIL_ASSIST_POD_RECORDS: tuple[dict[str, Any], ...] = (
    {
        "id": "bluerobotics_m200_24v",
        "vendor": "Blue Robotics",
        "family": "M200 Pod",
        "variant": "24 V bus",
        "topology": "waterproof brushless outrunner",
        "cooling": "water",
        "mass_kg": 0.44,
        "od_mm": 52.0,
        "price_usd": 210.0,
        "rpm_max": 4560.0,
        "tq_cont_nm": 0.65,
        "tq_peak_nm": 1.05,
        "p_cont_kw": 0.25,
        "p_peak_kw": 0.45,
        "v_dc": 24.0,
        "kt": 30.0 / (math.pi * 190.0),
        "kv": 190.0,
        "i_cont": 14.0,
        "i_peak": 22.0,
        "eff_includes_controller": False,
        "tier": "B",
        "eff": {
            "kind": "bldc",
            "kt": 30.0 / (math.pi * 190.0),
            "kv": 190.0,
            "r": 0.180,
            "i0": 0.65,
            "i_max": 22.0,
            "v_bus": 24.0,
        },
    },
    {
        "id": "flipsky_6354_140kv_24v",
        "vendor": "Flipsky",
        "family": "Flipsky 6354 140kv",
        "variant": "24 V bus",
        "topology": "radial-flux BLDC outrunner",
        "cooling": "water (IP-rated)",
        "mass_kg": 0.65,
        "od_mm": 63.0,
        "price_usd": 89.0,
        "rpm_max": 3528.0,
        "tq_cont_nm": 1.50,
        "tq_peak_nm": 2.80,
        "p_cont_kw": 0.35,
        "p_peak_kw": 0.70,
        "v_dc": 25.2,
        "kt": 30.0 / (math.pi * 140.0),
        "kv": 140.0,
        "i_cont": 24.0,
        "i_peak": 45.0,
        "eff_includes_controller": False,
        "tier": "B",
        "eff": {
            "kind": "bldc",
            "kt": 30.0 / (math.pi * 140.0),
            "kv": 140.0,
            "r": 0.140,
            "i0": 1.0,
            "i_max": 45.0,
            "v_bus": 25.2,
        },
    },
    {
        "id": "flipsky_6354_140kv_36v",
        "vendor": "Flipsky",
        "family": "Flipsky 6354 140kv",
        "variant": "36 V bus",
        "topology": "radial-flux BLDC outrunner",
        "cooling": "water (IP-rated)",
        "mass_kg": 0.65,
        "od_mm": 63.0,
        "price_usd": 89.0,
        "rpm_max": 5040.0,
        "tq_cont_nm": 1.50,
        "tq_peak_nm": 2.80,
        "p_cont_kw": 0.52,
        "p_peak_kw": 1.05,
        "v_dc": 36.0,
        "kt": 30.0 / (math.pi * 140.0),
        "kv": 140.0,
        "i_cont": 24.0,
        "i_peak": 45.0,
        "eff_includes_controller": False,
        "tier": "B",
        "eff": {
            "kind": "bldc",
            "kt": 30.0 / (math.pi * 140.0),
            "kv": 140.0,
            "r": 0.140,
            "i0": 1.0,
            "i_max": 45.0,
            "v_bus": 36.0,
        },
    },
    {
        "id": "flipsky_6384_120kv_36v",
        "vendor": "Flipsky",
        "family": "Flipsky 6384 120kv",
        "variant": "36 V bus",
        "topology": "radial-flux BLDC outrunner",
        "cooling": "water (IP-rated)",
        "mass_kg": 1.15,
        "od_mm": 63.0,
        "price_usd": 129.0,
        "rpm_max": 4320.0,
        "tq_cont_nm": 3.80,
        "tq_peak_nm": 6.20,
        "p_cont_kw": 0.95,
        "p_peak_kw": 1.90,
        "v_dc": 36.0,
        "kt": 30.0 / (math.pi * 120.0),
        "kv": 120.0,
        "i_cont": 50.0,
        "i_peak": 80.0,
        "eff_includes_controller": False,
        "tier": "B",
        "eff": {
            "kind": "bldc",
            "kt": 30.0 / (math.pi * 120.0),
            "kv": 120.0,
            "r": 0.075,
            "i0": 1.5,
            "i_max": 80.0,
            "v_bus": 36.0,
        },
    },
)


@dataclass
class CatalogMotor:
    """Structured wrapper around a motor catalog record with differentiable helpers."""

    id: str
    vendor: str
    family: str | None
    variant: str | None
    topology: str | None
    tier: str | None
    stack: float | None
    rpm_max: float | None
    tq_peak_nm: float | None
    tq_cont_nm: float | None
    p_peak_kw: float | None
    p_cont_kw: float | None
    v_dc: float | None
    kt: float | None
    kv: float | None
    kv_nom: float | None
    kv_peak: float | None
    i_cont: float | None
    i_peak: float | None
    i_dc_cont: float | None
    i_dc_peak: float | None
    eff_includes_controller: bool
    mass_kg: float | None
    od_mm: float | None
    price_usd: float | None
    eff: dict[str, Any] | None
    record: dict[str, Any]

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "CatalogMotor":
        return cls(
            id=rec["id"],
            vendor=rec["vendor"],
            family=rec.get("family"),
            variant=rec.get("variant"),
            topology=rec.get("topology"),
            tier=rec.get("tier"),
            stack=rec.get("stack"),
            rpm_max=rec.get("rpm_max"),
            tq_peak_nm=rec.get("tq_peak_nm"),
            tq_cont_nm=rec.get("tq_cont_nm"),
            p_peak_kw=rec.get("p_peak_kw"),
            p_cont_kw=rec.get("p_cont_kw"),
            v_dc=rec.get("v_dc"),
            kt=rec.get("kt"),
            kv=rec.get("kv"),
            kv_nom=rec.get("kv_nom"),
            kv_peak=rec.get("kv_peak"),
            i_cont=rec.get("i_cont"),
            i_peak=rec.get("i_peak"),
            i_dc_cont=rec.get("i_dc_cont"),
            i_dc_peak=rec.get("i_dc_peak"),
            eff_includes_controller=bool(rec.get("eff_includes_controller", False)),
            mass_kg=rec.get("mass_kg"),
            od_mm=rec.get("od_mm"),
            price_usd=rec.get("price_usd"),
            eff=rec.get("eff"),
            record=rec,
        )

    def performance_smooth(
        self,
        rpm: Vectorizable,
        torque: Vectorizable,
        controller_efficiency: float = CONTROLLER_ETA_DEFAULT,
        v_bus_override: Vectorizable | None = None,
    ) -> dict[str, Vectorizable]:
        """
        Smooth, CasADi-differentiable performance evaluation of this catalog motor.

        For ``kind == "bldc"``, evaluates the Drela/Flipsky circuit ``(kt, kv, r, i0)``.
        For ``kind in ("loss", "pmsm")``, evaluates the 4-term polynomial loss model.
        For ``kind == "map"``, evaluates its smooth ``fallback`` 4-term loss model.
        """
        if self.eff is None:
            raise ValueError(f"Motor {self.id!r} has no efficiency model.")

        kind = self.eff.get("kind")
        n_stack = float(self.stack) if (self.stack is not None and self.stack != 0.0) else 1.0
        v_bus = self.v_dc if v_bus_override is None else v_bus_override

        if kind == "bldc":
            e = self.eff
            omega = 2.0 * math.pi * rpm / 60.0
            qm = torque / n_stack
            p_shaft = torque * omega
            i_mot = qm / e["kt"] + e["i0"]
            v_req = rpm / e["kv"] + i_mot * e["r"]
            p_elec_one = np.softmax(v_req * i_mot, 1e-6, softness=1e-4)
            p_elec_motor = n_stack * p_elec_one
            eta_motor = p_shaft / np.fmax(p_elec_motor, 1e-9)
            ce = 1.0 if self.eff_includes_controller else controller_efficiency
            eta_drive = eta_motor * ce
            p_elec = p_elec_motor / np.fmax(ce, 1e-9)
            i_dc_pack = p_elec / np.fmax(v_bus if v_bus is not None else e["v_bus"], 1e-6)
            i_dc_min_bus = p_elec / np.fmax(v_req, 1e-6)
            return {
                "rpm": rpm,
                "omega": omega,
                "torque": torque,
                "P_shaft": p_shaft,
                "shaft_power": p_shaft,
                "P_loss_motor": p_elec_motor - p_shaft,
                "P_loss_controller": p_elec - p_elec_motor,
                "P_elec_motor": p_elec_motor,
                "P_elec": p_elec,
                "electrical_power": p_elec,
                "eta_motor": eta_motor,
                "efficiency": eta_motor,
                "eta_drive": eta_drive,
                "I_arms": i_mot,
                "V_req": v_req,
                "I_dc_pack": i_dc_pack,
                "I_dc_min_bus": i_dc_min_bus,
            }

        if kind == "pmsm":
            e = self.eff
            return motor_performance_loss(
                rpm=rpm,
                torque=torque,
                c_cu=e["c_cu"],
                k_h=e.get("k_h", 0.0),
                k_e=e.get("k_e", 0.0),
                k_0=e.get("k_0", 0.0),
                kt=e["kt"],
                kv=self.kv,
                r_ll=e["r_ll"],
                v_bus=v_bus,
                controller_efficiency=controller_efficiency,
                eff_includes_controller=self.eff_includes_controller,
                topology="pmsm",
                stack=int(n_stack),
            )

        if kind == "loss":
            e = self.eff
            return motor_performance_loss(
                rpm=rpm,
                torque=torque,
                c_cu=e["c_cu"],
                k_h=e.get("k_h", 0.0),
                k_e=e.get("k_e", 0.0),
                k_0=e.get("k_0", 0.0),
                kt=self.kt,
                kv=self.kv,
                r_ll=float(self.record.get("r_phase", 0.0) or 0.0),
                v_bus=v_bus,
                controller_efficiency=controller_efficiency,
                eff_includes_controller=self.eff_includes_controller,
                topology="loss",
                stack=int(n_stack),
            )

        if kind == "map":
            fb = self.eff.get("fallback")
            if fb is None:
                raise ValueError(f"Map motor {self.id!r} lacks a smooth fallback loss model.")
            return motor_performance_loss(
                rpm=rpm,
                torque=torque,
                c_cu=fb["c_cu"],
                k_h=fb.get("k_h", 0.0),
                k_e=fb.get("k_e", 0.0),
                k_0=fb.get("k_0", 0.0),
                kt=self.kt,
                kv=self.kv,
                r_ll=float(self.record.get("r_phase", 0.0) or 0.0),
                v_bus=v_bus,
                controller_efficiency=controller_efficiency,
                eff_includes_controller=self.eff_includes_controller,
                topology="loss",
                stack=int(n_stack),
            )

        raise ValueError(f"Unsupported efficiency model kind {kind!r} on motor {self.id!r}.")

    def envelope_margins(
        self,
        rpm: Vectorizable,
        torque: Vectorizable,
        rating: Literal["cont", "peak"] = "cont",
        controller_efficiency: float = CONTROLLER_ETA_DEFAULT,
        v_bus_override: Vectorizable | None = None,
    ) -> dict[str, Vectorizable]:
        """
        Return normalized operating-envelope margins ``(limit - value) / limit``
        (all ``>= 0`` when within ratings) suitable as smooth ``Opti`` constraints.
        """
        perf = self.performance_smooth(
            rpm=rpm,
            torque=torque,
            controller_efficiency=controller_efficiency,
            v_bus_override=v_bus_override,
        )
        margins: dict[str, Vectorizable] = {}
        is_peak = rating == "peak"

        if self.rpm_max is not None and self.rpm_max > 0:
            margins["rpm"] = (self.rpm_max - rpm) / self.rpm_max

        tq_lim = self.tq_peak_nm if is_peak else self.tq_cont_nm
        if tq_lim is not None and tq_lim > 0:
            margins["torque"] = (tq_lim - torque) / tq_lim

        p_lim_kw = self.p_peak_kw if is_peak else self.p_cont_kw
        if p_lim_kw is not None and p_lim_kw > 0:
            p_lim_w = 1000.0 * p_lim_kw
            margins["power"] = (p_lim_w - perf["P_shaft"]) / p_lim_w

        if self.eff is not None and self.eff.get("kind") == "bldc":
            i_lim = self.eff.get("i_max") if is_peak else (self.i_cont or self.eff.get("i_max"))
            if i_lim is not None and i_lim > 0 and perf["I_arms"] is not None:
                margins["ac_current"] = (i_lim - perf["I_arms"]) / i_lim
            v_lim = self.eff.get("v_bus") if v_bus_override is None else v_bus_override
            if v_lim is not None and perf["V_req"] is not None:
                margins["bus_voltage"] = (v_lim - perf["V_req"]) / v_lim
        else:
            if self.eff is not None and self.eff.get("kind") == "pmsm":
                i_lim = self.eff["i_peak"] if is_peak else self.eff["i_cont"]
            else:
                i_lim = self.i_peak if is_peak else self.i_cont
            if i_lim is not None and i_lim > 0 and perf["I_arms"] is not None:
                margins["ac_current"] = (i_lim - perf["I_arms"]) / i_lim
            v_lim = self.v_dc if v_bus_override is None else v_bus_override
            if v_lim is not None and perf["V_req"] is not None:
                margins["bus_voltage"] = (v_lim - perf["V_req"]) / v_lim

        i_dc_lim = self.i_dc_peak if is_peak else self.i_dc_cont
        if i_dc_lim is not None and i_dc_lim > 0 and perf["I_dc_pack"] is not None:
            margins["dc_current"] = (i_dc_lim - perf["I_dc_pack"]) / i_dc_lim

        return margins


class MotorDatabase:
    """In-memory database of electric motors and digitized efficiency maps."""

    def __init__(
        self,
        motors: list[CatalogMotor],
        maps: dict[str, dict[str, Any]],
        controller_eta: float = CONTROLLER_ETA_DEFAULT,
        raw: dict[str, Any] | None = None,
    ):
        self.motors = motors
        self.maps = maps
        self.controller_eta = float(controller_eta)
        self.raw = raw or {}
        self._by_id = {m.id: m for m in motors}

    @classmethod
    def vendored(cls, include_foil_assist_pods: bool = False) -> "MotorDatabase":
        """Load the compressed vendored motor catalog."""
        raw = json.loads(gzip.decompress(_DEFAULT_DB_PATH.read_bytes()).decode("utf-8"))
        return cls.from_dict(raw, include_foil_assist_pods=include_foil_assist_pods)

    @classmethod
    def from_boatmath_js(
        cls,
        path: str | Path,
        include_foil_assist_pods: bool = False,
    ) -> "MotorDatabase":
        """Load directly from ``boatmath`` / ``propopt`` ``motors.js``."""
        text = Path(path).read_text(encoding="utf-8")
        raw, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
        return cls.from_dict(raw, include_foil_assist_pods=include_foil_assist_pods)

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        include_foil_assist_pods: bool = False,
    ) -> "MotorDatabase":
        motors = [CatalogMotor.from_record(m) for m in raw["motors"]]
        if include_foil_assist_pods:
            existing = {m.id for m in motors}
            for rec in _FOIL_ASSIST_POD_RECORDS:
                if rec["id"] not in existing:
                    motors.append(CatalogMotor.from_record(dict(rec)))
        return cls(
            motors=motors,
            maps=dict(raw.get("maps", {})),
            controller_eta=raw.get("controller_eta", 1.0),
            raw=raw,
        )

    def motor(self, motor_id: str) -> CatalogMotor | None:
        return self._by_id.get(motor_id)

    get = motor


def _loss_at(loss: dict[str, Any], q: float, w: float) -> float:
    return (
        loss["c_cu"] * q * q
        + loss.get("k_h", 0.0) * w
        + loss.get("k_e", 0.0) * w * w
        + loss.get("k_0", 0.0)
    )


def motor_bus_voltage(
    m: CatalogMotor, rpm: float, i_arms: float | None = None
) -> float | None:
    """
    Estimate required DC bus voltage [V] at ``(rpm, i_arms)`` matching
    ``boatmath::propeller::motor::bus_voltage``.
    """
    eff = m.eff
    if eff is not None:
        kind = eff.get("kind")
        if kind == "pmsm":
            kb_hot = eff["kt"] / _KB_HOT_FACTOR
            v_ll = kb_hot * rpm / 1000.0 + (i_arms or 0.0) * eff["r_ll"]
            return math.sqrt(2.0) * v_ll
        if kind == "bldc":
            return rpm / eff["kv"] + (i_arms or 0.0) * eff["r"]

    if m.kv is None or m.kv == 0.0:
        return None
    kv = float(m.kv)
    if (
        m.kv_nom is not None
        and m.i_cont is not None
        and m.i_cont != 0.0
        and i_arms is not None
    ):
        if i_arms <= m.i_cont:
            kv = kv + (m.kv_nom - kv) * (i_arms / m.i_cont)
        elif m.kv_peak is not None and m.i_peak is not None and m.i_peak > m.i_cont:
            f = min((i_arms - m.i_cont) / (m.i_peak - m.i_cont), 1.0)
            kv = m.kv_nom + (m.kv_peak - m.kv_nom) * f
        else:
            kv = m.kv_nom
    return (rpm / kv) if kv > 0.0 else None


def _sample_eff_map(eff_map: dict[str, Any], rpm: float, q: float) -> tuple[float, bool] | None:
    r0, r1, nr = eff_map["rpm"]
    q0, q1, nq = eff_map["tq"]
    if rpm < r0 or rpm > r1 or q < q0 or q > q1:
        return None
    nr, nq = int(nr), int(nq)
    fr = (rpm - r0) / (r1 - r0) * (nr - 1)
    fq = (q - q0) / (q1 - q0) * (nq - 1)
    i0 = min(int(max(math.floor(fr), 0.0)), nr - 2)
    j0 = min(int(max(math.floor(fq), 0.0)), nq - 2)
    tr, tq = fr - i0, fq - j0
    eta_grid = eff_map["eta"]
    inside_grid = eff_map["inside"]
    at = lambda i, j: eta_grid[i * nq + j] / 10000.0
    inside = lambda i, j: inside_grid[i * nq + j] == 1
    e = (
        at(i0, j0) * (1.0 - tr) * (1.0 - tq)
        + at(i0 + 1, j0) * tr * (1.0 - tq)
        + at(i0, j0 + 1) * (1.0 - tr) * tq
        + at(i0 + 1, j0 + 1) * tr * tq
    )
    solid = inside(i0, j0) and inside(i0 + 1, j0) and inside(i0, j0 + 1) and inside(i0 + 1, j0 + 1)
    return e, (not solid)


def _finish_point(
    m: CatalogMotor,
    rpm: float,
    q: float,
    qm: float,
    p_shaft: float,
    eta: float,
    extrapolated: bool,
    rating: Literal["cont", "peak"],
    controller_eta: float,
) -> dict[str, Any]:
    i_arms = (qm / m.kt) if (m.kt is not None and m.kt != 0.0) else None
    v_bus = motor_bus_voltage(m, rpm, i_arms)
    if v_bus is not None and m.v_dc is not None and m.v_dc != 0.0 and v_bus > m.v_dc * SLACK:
        return {"ok": False, "limit": "voltage"}

    ce = controller_eta if controller_eta != 0.0 else 1.0
    eta_drv = eta if m.eff_includes_controller else eta * ce
    p_elec = p_shaft / eta_drv
    v_for_i = (
        v_bus
        if (v_bus is not None and v_bus > 0.0)
        else (m.v_dc if (m.v_dc is not None and m.v_dc != 0.0) else None)
    )
    i_dc = (p_elec / v_for_i) if v_for_i is not None else None
    if i_dc is not None:
        lim = m.i_dc_peak if rating == "peak" else m.i_dc_cont
        if lim is not None and lim != 0.0 and i_dc > lim * SLACK:
            return {"ok": False, "limit": "dc-current"}

    i_dc_pack = (p_elec / m.v_dc) if (m.v_dc is not None and m.v_dc > 0.0) else i_dc
    return {
        "ok": True,
        "eta": eta,
        "etaWithController": eta_drv,
        "extrapolated": extrapolated,
        "rpm": rpm,
        "Q": q,
        "P_shaft": p_shaft,
        "P_elec": p_elec,
        "I_arms": i_arms,
        "V_bus": v_bus,
        "I_dc": i_dc,
        "I_dc_pack": i_dc_pack,
        "ratio": 1.0,
        "gearEta": 1.0,
        "etaDrive": eta_drv,
        "top": None,
    }


def evaluate_catalog_motor(
    db: MotorDatabase,
    motor: CatalogMotor,
    rpm: float,
    torque: float,
    rating: Literal["cont", "peak"] = "cont",
    controller_eta: float = 1.0,
) -> dict[str, Any] | None:
    """
    Evaluate a catalog motor at motor shaft ``(rpm, torque)`` including all
    published envelope limits (``speed``, ``torque``, ``power``, ``current``,
    ``voltage``, ``dc-current``).
    """
    if not (rpm >= 0.0 and torque > 0.0):
        return None
    if motor.rpm_max is not None and rpm > motor.rpm_max * SLACK:
        return {"ok": False, "limit": "speed"}
    tq_lim = motor.tq_peak_nm if rating == "peak" else motor.tq_cont_nm
    if tq_lim is not None and torque > tq_lim * SLACK:
        return {"ok": False, "limit": "torque"}

    w = 2.0 * math.pi * rpm / 60.0
    p_shaft = torque * w
    p_lim = motor.p_peak_kw if rating == "peak" else motor.p_cont_kw
    if p_lim is not None and p_shaft > 1000.0 * p_lim * SLACK:
        return {"ok": False, "limit": "power"}

    n = float(motor.stack) if (motor.stack is not None and motor.stack != 0.0) else 1.0
    qm = torque / n
    p_one = qm * w
    eff_spec = motor.eff
    if eff_spec is None:
        return {"ok": False, "limit": "no-efficiency-data"}

    extrapolated = False
    kind = eff_spec.get("kind")
    if kind == "map":
        mp = db.maps.get(eff_spec["map"])
        sample = _sample_eff_map(mp, rpm, qm) if mp is not None else None
        if sample is not None and not sample[1]:
            se = sample[0]
            p_loss_one = p_one * (1.0 / se - 1.0)
            p_loss = n * (p_loss_one + (eff_spec["ccu"] - eff_spec["ccu_map"]) * qm * qm)
            eta = p_shaft / (p_shaft + max(p_loss, 1e-9))
        else:
            fb = eff_spec.get("fallback")
            if fb is None:
                return {"ok": False, "limit": "outside-map"}
            p_loss = n * _loss_at(fb, qm, w)
            et = p_shaft / (p_shaft + max(p_loss, 1e-9))
            if not (0.0 < et < 1.0):
                return {"ok": False, "limit": "outside-map"}
            return _finish_point(motor, rpm, torque, qm, p_shaft, et, True, rating, controller_eta)
    elif kind == "bldc":
        i = qm / eff_spec["kt"] + eff_spec["i0"]
        if i > eff_spec["i_max"] * SLACK:
            return {"ok": False, "limit": "current"}
        v_req = rpm / eff_spec["kv"] + i * eff_spec["r"]
        if v_req > eff_spec["v_bus"] * SLACK:
            return {"ok": False, "limit": "voltage"}
        eta = p_one / (v_req * i)
    elif kind == "pmsm":
        i = qm / eff_spec["kt"]
        lim = eff_spec["i_peak"] if rating == "peak" else eff_spec["i_cont"]
        if i > lim * SLACK:
            return {"ok": False, "limit": "current"}
        p_loss = n * _loss_at(eff_spec, qm, w)
        eta = p_shaft / (p_shaft + max(p_loss, 1e-9))
    elif kind == "loss":
        p_loss = n * _loss_at(eff_spec, qm, w)
        eta = p_shaft / (p_shaft + max(p_loss, 1e-9))
    else:
        return {"ok": False, "limit": "no-efficiency-data"}

    if not (0.0 < eta < 1.0):
        return {"ok": False, "limit": "model"}
    return _finish_point(
        motor, rpm, torque, qm, p_shaft, eta, extrapolated, rating, controller_eta
    )


def through_gearbox(
    db: MotorDatabase,
    motor: CatalogMotor,
    prop_rpm: float,
    prop_torque: float,
    ratio: float,
    gear_eta: float = GEAR_ETA_DEFAULT,
    rating: Literal["cont", "peak"] = "cont",
    controller_eta: float = CONTROLLER_ETA_DEFAULT,
    second_point: tuple[float, float] | None = None,
    second_rating: Literal["cont", "peak"] = "peak",
) -> dict[str, Any] | None:
    """
    Evaluate a catalog motor driving a propeller load ``(prop_rpm, prop_torque)``
    through a reduction gearbox ``ratio`` (``motor_rpm = prop_rpm * ratio``,
    ``motor_torque = prop_torque / (ratio * eff_gear)``).
    """
    g = 1.0 if ratio <= 0.0 else float(ratio)
    eff_g = 1.0 if g == 1.0 else float(gear_eta)
    r = evaluate_catalog_motor(
        db,
        motor,
        prop_rpm * g,
        prop_torque / (g * eff_g),
        rating=rating,
        controller_eta=controller_eta,
    )
    if r is None or not r["ok"]:
        return r

    top_res = None
    if second_point is not None:
        s_rpm, s_q = second_point
        t = evaluate_catalog_motor(
            db,
            motor,
            s_rpm * g,
            s_q / (g * eff_g),
            rating=second_rating,
            controller_eta=controller_eta,
        )
        if t is None:
            return {"ok": False, "limit": "top-speed unreachable"}
        if not t["ok"]:
            return {"ok": False, "limit": f"top-speed {t['limit']}"}
        top_res = {
            "rpm": t["rpm"],
            "Q": t["Q"],
            "eta": t["eta"],
            "P_elec": t["P_elec"],
            "V_bus": t["V_bus"],
            "I_dc": t["I_dc"],
            "I_dc_pack": t["I_dc_pack"],
            "I_arms": t["I_arms"],
            "shaftRpm": s_rpm,
        }

    out = dict(r)
    out["ratio"] = g
    out["gearEta"] = eff_g
    out["top"] = top_res
    out["etaDrive"] = (prop_torque * 2.0 * math.pi * prop_rpm / 60.0) / r["P_elec"]
    return out


def best_gearbox_ratio(
    db: MotorDatabase,
    motor: CatalogMotor,
    prop_rpm: float,
    prop_torque: float,
    ratio_min: float = 1.0,
    ratio_max: float = 25.0,
    gear_eta: float = GEAR_ETA_DEFAULT,
    rating: Literal["cont", "peak"] = "cont",
    controller_eta: float = CONTROLLER_ETA_DEFAULT,
    second_point: tuple[float, float] | None = None,
    second_rating: Literal["cont", "peak"] = "peak",
    coarse: bool = False,
) -> dict[str, Any] | None:
    """Find the reduction ratio in ``[ratio_min, ratio_max]`` minimizing ``P_elec``."""
    nn = 26 if coarse else 80
    best = None
    for i in range(nn):
        g = ratio_min * (ratio_max / ratio_min) ** (i / (nn - 1))
        r = through_gearbox(
            db,
            motor,
            prop_rpm,
            prop_torque,
            g,
            gear_eta=gear_eta,
            rating=rating,
            controller_eta=controller_eta,
            second_point=second_point,
            second_rating=second_rating,
        )
        if r is not None and r["ok"] and (best is None or r["P_elec"] < best["P_elec"]):
            best = r
    if best is None or coarse:
        return best

    phi = 0.6180339887
    a = max(ratio_min, best["ratio"] / 1.25)
    b = min(ratio_max, best["ratio"] * 1.25)

    def eval_ratio(g: float):
        res = through_gearbox(
            db,
            motor,
            prop_rpm,
            prop_torque,
            g,
            gear_eta=gear_eta,
            rating=rating,
            controller_eta=controller_eta,
            second_point=second_point,
            second_rating=second_rating,
        )
        return res if (res is not None and res["ok"]) else None

    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)
    f1 = eval_ratio(x1)
    f2 = eval_ratio(x2)
    i = 0
    while i < 30 and (b - a) > 1e-4:
        p1 = f1["P_elec"] if f1 is not None else math.inf
        p2 = f2["P_elec"] if f2 is not None else math.inf
        if p1 < p2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - phi * (b - a)
            f1 = eval_ratio(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + phi * (b - a)
            f2 = eval_ratio(x2)
        i += 1
    for cand in (f1, f2):
        if cand is not None and cand["P_elec"] < best["P_elec"]:
            best = cand
    return best


def match_motors_to_curve(
    db: MotorDatabase,
    curve_points: Sequence[dict[str, Any]],
    *,
    rating: Literal["cont", "peak"] = "cont",
    gear_eta: float = GEAR_ETA_DEFAULT,
    controller_eta: float | None = None,
    ratio: float | None = None,
    ratio_min: float = 1.0,
    ratio_max: float = 25.0,
    check_top: bool = False,
    second_rating: Literal["cont", "peak"] = "peak",
    rank_by: Literal["power", "mass", "price"] = "power",
    all_windings: bool = False,
    single_only: bool = False,
    include: Callable[[CatalogMotor], bool] | None = None,
) -> dict[str, Any]:
    """
    Find each catalog motor's lowest-``P_elec`` point along a propeller curve
    (matching ``boatmath::propeller::motor::scatter``).
    """
    ce = db.controller_eta if controller_eta is None else float(controller_eta)
    feasible = [
        p
        for p in curve_points
        if p.get("ok")
        and p.get("design") is not None
        and (not check_top or p["design"].get("topOk"))
    ]
    if not feasible:
        return {"dots": [], "all": [], "unreachable": []}

    stride = max(len(feasible) // 48, 1)

    def solve(m: CatalogMotor, pt: dict[str, Any], coarse: bool) -> dict[str, Any] | None:
        d = pt["design"]
        sec = None
        if check_top and d.get("top") is not None:
            sec = (d["top"]["rpm"], d["top"]["Q"])
        if ratio is not None:
            res = through_gearbox(
                db,
                m,
                d["rpm"],
                d["Q"],
                ratio,
                gear_eta=gear_eta,
                rating=rating,
                controller_eta=ce,
                second_point=sec,
                second_rating=second_rating,
            )
            return res if (res is not None and res["ok"]) else None
        return best_gearbox_ratio(
            db,
            m,
            d["rpm"],
            d["Q"],
            ratio_min=ratio_min,
            ratio_max=ratio_max,
            gear_eta=gear_eta,
            rating=rating,
            controller_eta=ce,
            second_point=sec,
            second_rating=second_rating,
            coarse=coarse,
        )

    dots: list[dict[str, Any]] = []
    unreachable: list[CatalogMotor] = []
    for m in db.motors:
        if single_only and m.stack is not None and m.stack > 1.0:
            continue
        if include is not None and not include(m):
            continue

        bi = None
        bp = None
        i = 0
        while i < len(feasible):
            r = solve(m, feasible[i], True)
            if r is not None and (bp is None or r["P_elec"] < bp):
                bp = r["P_elec"]
                bi = i
            i += stride
        if bi is None:
            unreachable.append(m)
            continue

        lo = max(0, bi - stride)
        hi = min(len(feasible) - 1, bi + stride)
        best_pair: tuple[dict[str, Any], int] | None = None
        for k in range(lo, hi + 1):
            r = solve(m, feasible[k], False)
            if r is not None and (best_pair is None or r["P_elec"] < best_pair[0]["P_elec"]):
                best_pair = (r, k)
        if best_pair is None:
            unreachable.append(m)
            continue

        run, k = best_pair
        load = (
            run["Q"] / m.tq_cont_nm
            if (m.tq_cont_nm is not None and m.tq_cont_nm != 0.0)
            else None
        )
        dots.append(
            {
                "motor": m,
                "point": feasible[k],
                "run": run,
                "load": load,
            }
        )

    def sort_key(d: dict[str, Any]):
        p_el = d["run"]["P_elec"]
        if rank_by == "mass":
            v = d["motor"].mass_kg
            return (1 if v is None else 0, 0.0 if v is None else v, p_el)
        if rank_by == "price":
            v = d["motor"].price_usd
            return (1 if v is None else 0, 0.0 if v is None else v, p_el)
        return (0, 0.0, p_el)

    dots.sort(key=sort_key)
    if all_windings:
        shown = list(dots)
    else:
        seen: list[str | None] = []
        shown = []
        for d in dots:
            fam = d["motor"].family
            if fam not in seen:
                seen.append(fam)
                shown.append(d)
        shown.sort(key=sort_key)

    return {"dots": shown, "all": dots, "unreachable": unreachable}
