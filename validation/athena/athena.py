"""
R/V Athena (DTNSRDC Model 5365), a high-speed transom-stern hull, as a `Hull`.

Offsets from Jenkins (1984), "Resistance Characteristics of the High Speed
Transom Stern Ship R/V Athena in the Bare Hull Condition", DTNSRDC-84/024,
Table 2: half-breadths at seven waterlines (0.125 H .. 1.5 H above the
baseline), each as a fraction of that waterline's greatest half-breadth, which
is given as a fraction of B_max / 2. H is the draught. Lpp 5.690 m, H 0.183 m,
B_max 0.836 m, wetted surface 4.222 m^2, C_B 0.4775 (Table 1).

The table does not give the keel line. Forward of station 14 a section is
taken to reach the centreline at the highest waterline where the table has
zero; aft of it the centreline height is interpolated to the buttock heights
of Table 3 (stations 18 to 20, drawing without wedge).
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from hydrosandbox.hull import Hull

LPP, H, B_MAX, WETTED_SURFACE = 5.690, 0.183, 0.836, 4.222
CB_REPORT = 0.4775
WATERLINES = np.array([0.125, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50])  # z / H
MAX_HALF_BEAM = np.array([0.3538, 0.5431, 0.7937, 0.9424, 1.0000, 1.0170, 1.0303])  # / (B_max / 2)
OFFSETS = {  # station (of 20, from the forward perpendicular): half-breadth / waterline maximum
    0.0: [0.0000, 0.0000, 0.0000, 0.0000, 0.0048, 0.0185, 0.0347],
    0.5: [0.0000, 0.0246, 0.0359, 0.0451, 0.0570, 0.0758, 0.0989],
    1.0: [0.0000, 0.0525, 0.0818, 0.0959, 0.1110, 0.1354, 0.1637],
    1.5: [0.0000, 0.0838, 0.1292, 0.1462, 0.1675, 0.1945, 0.2267],
    2.0: [0.0000, 0.1162, 0.1766, 0.2035, 0.2257, 0.2542, 0.2886],
    3.0: [0.0377, 0.1955, 0.2813, 0.3104, 0.3398, 0.3711, 0.4081],
    4.0: [0.1029, 0.2849, 0.3891, 0.4218, 0.4478, 0.4761, 0.5129],
    5.0: [0.1972, 0.3989, 0.4992, 0.5280, 0.5643, 0.5776, 0.6078],
    6.0: [0.3036, 0.4972, 0.6009, 0.6246, 0.6462, 0.6700, 0.6990],
    7.0: [0.4305, 0.6190, 0.6934, 0.7070, 0.7263, 0.7476, 0.7703],
    8.0: [0.5918, 0.7262, 0.7783, 0.7830, 0.7967, 0.8156, 0.8345],
    9.0: [0.7410, 0.8346, 0.8517, 0.8448, 0.8568, 0.8807, 0.8869],
    10.0: [0.8868, 0.9240, 0.9136, 0.9002, 0.9065, 0.9177, 0.9199],
    11.0: [1.0000, 1.0000, 0.9671, 0.9420, 0.9381, 0.9457, 0.9488],
    12.0: [0.8353, 0.9519, 1.0000, 0.9762, 0.9660, 0.9684, 0.9699],
    13.0: [0.4580, 0.8424, 1.0000, 0.9942, 0.9872, 0.9875, 0.9888],
    14.0: [0.0000, 0.5765, 0.9801, 1.0000, 1.0000, 1.0000, 1.0000],
    15.0: [0.0000, 0.0581, 0.9113, 0.9865, 0.9939, 0.9946, 0.9953],
    16.0: [0.0000, 0.0000, 0.7645, 0.9575, 0.9751, 0.9791, 0.9799],
    17.0: [0.0000, 0.0000, 0.4870, 0.9227, 0.9478, 0.9505, 0.9517],
    18.0: [0.0000, 0.0000, 0.0871, 0.8731, 0.9108, 0.9147, 0.9134],
    18.5: [0.0000, 0.0000, 0.0000, 0.8545, 0.8926, 0.8920, 0.8899],
    19.0: [0.0000, 0.0000, 0.0000, 0.8345, 0.8695, 0.8711, 0.8669],
    19.5: [0.0000, 0.0000, 0.0000, 0.8068, 0.8477, 0.8461, 0.8416],
    20.0: [0.0000, 0.0000, 0.0000, 0.8023, 0.8289, 0.8216, 0.8168],
}
# Centreline (keel) height / H where the table does not settle it.
KEEL = {18.0: 0.482, 19.0: 0.565, 19.5: 0.613, 20.0: 0.638}  # Table 3
KEEL.update({14.0: 0.125, 18.5: 0.5 * (0.482 + 0.565)})
for s in (15.0, 16.0, 17.0):  # a straight buttock from station 14 to 18
    KEEL[s] = 0.125 + (0.482 - 0.125) * (s - 14.0) / 4.0


def athena_hull(n_stations: int = 81, n_levels: int = 49, **hull_options) -> Hull:
    z = np.linspace(0.0, 1.5, n_levels)  # / H
    stations = np.array(sorted(OFFSETS))
    table = np.zeros((len(stations), n_levels))
    for i, s in enumerate(stations):
        y = np.array(OFFSETS[s]) * MAX_HALF_BEAM
        wet = y > 0
        if s in KEEL:
            keel = KEEL[s]
            wet &= WATERLINES > keel + 1e-6
        else:  # the highest waterline at which the table is still zero (the baseline if none)
            keel = WATERLINES[~wet].max() if (~wet).any() else 0.0
        zs = np.concatenate([[keel], WATERLINES[wet]])
        ys = np.concatenate([[0.0], y[wet]])
        table[i] = np.where(z >= keel, np.clip(PchipInterpolator(zs, ys)(np.clip(z, keel, 1.5)), 0.0, None), 0.0)
    x = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n_stations)))
    half_breadths = np.array([PchipInterpolator(stations / 20.0, table[:, j])(x) for j in range(n_levels)]).T
    return Hull(x * LPP, (z - 1.0) * H, np.clip(half_breadths, 0.0, None) * B_MAX / 2,
                name="R/V Athena, Model 5365", **hull_options)


if __name__ == "__main__":
    hull = athena_hull()
    hs = hull.hydrostatics()
    volume, surface, x_cb = (float(hs[k]) for k in ("volume", "wetted_surface", "x_cb"))
    b_mid = 0.1332 * LPP
    print(f"C_B on midship beam {volume / (LPP * b_mid * H):.4f} (report {CB_REPORT})")
    print(f"wetted surface {surface:.3f} m^2 (report {WETTED_SURFACE}); LCB {x_cb / LPP:.3f} L from FP; transom {hull.has_transom}")
