"""
Trace the Series 64 model 5s body plan (Wellicome, Molland, Cic & Taunton 1999,
Ship Science Report 106, Fig. 1) from a rendered page image into normalised
sections, `sections_5s.json`.

    uv run --with pillow python validation/series64/trace_bodyplan.py bodyplan106.png

The image is the figure rendered at 5x (page 11 of the report, clipped to
27-72% of the width and 50-82% of the height). The forebody (stations 1-10) is
on the right, the afterbody (11-20) on the left; station 0 is the stem. Curves
are followed by scanning rows (sides) and columns (the afterbody's flat
bottoms) and counting line crossings outward from the centreline; where
neighbouring afterbody curves merge into one dark run they are spread evenly
across it; forebody rows where curves merge (near the keel) are skipped.
Output: half-breadth / (B/2) against height above base / T, stations 0-20.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

dark = np.array(Image.open(sys.argv[1]).convert("L")) < 128
XC = 655  # centreline column
SIDES = {"fore": dict(sign=+1, base=1288, dwl=881), "aft": dict(sign=-1, base=1284, dwl=876)}
LINE = 5  # line thickness [px]


def runs(line):
    d = np.diff(np.concatenate([[0], line.astype(int), [0]]))
    return [(int(a), int(b)) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def split(found, expected):
    """Centres of `expected` curves from the dark runs found (merged runs hold several)."""
    if len(found) > expected or not found:
        return None
    holds = [1] * len(found)
    while sum(holds) < expected:  # give the spare curves to the runs with most width per curve
        k = int(np.argmax([(b - a) / h for (a, b), h in zip(found, holds)]))
        if (found[k][1] - found[k][0]) / holds[k] < 0.8 * LINE:
            return None
        holds[k] += 1
    centres = []
    for (a, b), h in zip(found, holds):
        centres += list(np.linspace(a + LINE / 2, b - LINE / 2, h)) if h > 1 else [0.5 * (a + b)]
    return centres


points = {k: [] for k in range(1, 21)}  # station -> (y px from centreline, z px above base)

### Forebody: every row has all ten curves, station 10 outermost.
s = SIDES["fore"]
for row in range(s["base"] - 4, s["base"] - int(1.75 * (s["base"] - s["dwl"])), -1):
    if abs(row - s["dwl"]) < 28:
        continue  # the waterline mark and its symbol outboard
    found = [(a + 3, b + 3) for a, b in runs(dark[row, XC + 3:XC + 600]) if not (a == 0 and b - a <= 4)]
    if len(found) == 10:  # all ten curves separate (they merge near the keel: filled in below)
        for i, (a, b) in enumerate(found):
            points[1 + i].append((0.5 * (a + b), s["base"] - row))

### Afterbody bottoms, by column: the highest line is the transom (station 20).
s = SIDES["aft"]
for col in range(XC - 12, XC - 330, -2):
    found = [(a + 1000, b + 1000) for a, b in runs(dark[1000:s["base"] - 4, col])]
    if len(found) == 9 and min(b - a for a, b in found) >= 1 and max(b - a for a, b in found) <= 8:
        for i, (a, b) in enumerate(found):
            points[20 - i].append((XC - col, s["base"] - 0.5 * (a + b)))
bottom = {k: min(z for _, z in points[k]) for k in range(12, 21)}
bottom[11] = 0.0
points[11].append((0.0, 0.0))

### Afterbody sides, by row: the curves present are the stations whose bottom lies below the row.
for row in range(s["base"] - 6, s["base"] - int(1.75 * (s["base"] - s["dwl"])), -1):
    if abs(row - s["dwl"]) < 28:
        continue
    z = s["base"] - row
    present = [k for k in range(11, 21) if bottom[k] + 12 < z]
    found = [(a, b) for a, b in runs(dark[row, :XC - 12][::-1]) if 150 < a and b < 548]  # outboard of the flat bottoms, inboard of the page edge
    found = [(a + 12, b + 12) for a, b in found]
    centres = split(found, len(present)) if present else None
    if centres:
        for k, y in zip(sorted(present, reverse=True), sorted(centres)):  # the transom is innermost
            points[k].append((y, z))

T_px = 0.5 * sum(v["base"] - v["dwl"] for v in SIDES.values())
z_grid = np.linspace(0.0, 1.7, 171)
table = [np.zeros_like(z_grid)]
for k in range(1, 21):
    p = np.array(sorted(points[k], key=lambda q: (q[1], q[0])))
    z, y = p[:, 1] / T_px, np.maximum.accumulate(p[:, 0])
    keep = np.concatenate([[True], np.diff(z) > 1e-9])
    hb = np.interp(z_grid, z[keep], y[keep], left=0.0)
    if k <= 10:  # forebody below the lowest clean row: a parabola from the keel
        hb = np.where(z_grid < z[0], y[0] * np.sqrt(z_grid / z[0]), hb)
    else:
        hb = np.where(z_grid < z[0] - 1e-9, 0.0, hb)
    table.append(hb)
table = np.array(table)
half_beam = max(np.interp(1.0, z_grid, row) for row in table)
out = dict(z_over_T=[round(float(v), 4) for v in z_grid],
           half_breadth_over_half_beam=[[round(float(v), 4) for v in row / half_beam] for row in table])
(Path(__file__).parent / "sections_5s.json").write_text(json.dumps(out))
print("points per station:", {k: len(v) for k, v in points.items()})
print("bottom heights / T:", {k: round(v / T_px, 3) for k, v in bottom.items()})
print("DWL half-breadth / max:", [round(float(np.interp(1.0, z_grid, row) / half_beam), 3) for row in table])
print("image B/T:", round(2 * half_beam / T_px, 3))
