"""
The camber port (``hydrosandbox.hull.camber``) against camber itself, and camber
hulls and hull families as ``Hull``s.

``tests/camber/golden.json`` is camber's own output for the documents in
``tests/camber/documents`` (see ``tests/camber/make_golden.mjs``).
"""

import json
from pathlib import Path

import numpy as onp
import pytest
import aerosandbox as asb

from hydrosandbox.hull import Hull
from hydrosandbox.hull.camber import (
    CamberHull,
    HullFamily,
    SweptHull,
    blend,
    parse_document,
    promote_family,
)
from hydrosandbox.hull.camber.blended import lattice_basis, simplex_lattice
from hydrosandbox.hull.camber.curves import centripetal_params, cr_chain, knuckle_slopes, pchip_slopes

HERE = Path(__file__).parent / "camber"
GOLDEN = json.loads((HERE / "golden.json").read_text())
NAMES = sorted(GOLDEN["documents"])


def load(name: str) -> CamberHull:
    return parse_document(json.loads((HERE / "documents" / f"{name}.json").read_text()))


def flat(doc: dict) -> onp.ndarray:
    """Every authored number of a version-2 document, in order."""
    v = [c for p in doc["sheerPlan"] for c in (p["x"], p["y"])]
    v += [c for p in doc["sheerTrim"] for c in (p["x"], p["z"], p["k"])]
    v += [c for p in doc["transom"] for c in (p["x"], p["z"])]
    for s in doc["stations"]:
        v += [s["u"], s["keelK"]] + [q for p in s["points"] for q in (p["n"], p["z"], p["k"])]
    return onp.array(v, dtype=float)


def test_primitives_match_camber():
    p = GOLDEN["primitives"]
    assert onp.allclose(pchip_slopes(p["pchip"]["xs"], p["pchip"]["ys"]), p["pchip"]["m"], atol=1e-14)
    left, right = knuckle_slopes(p["knuckle"]["xs"], p["knuckle"]["ys"], p["knuckle"]["ks"])
    assert onp.allclose(left, p["knuckle"]["L"], atol=1e-14)
    assert onp.allclose(right, p["knuckle"]["R"], atol=1e-14)
    c = p["crChain"]
    assert onp.allclose(centripetal_params(c["pts"]), c["t"], atol=1e-14)
    assert onp.allclose(cr_chain(c["pts"], c["t"], c["ks"]), c["chain"], atol=1e-14)


@pytest.mark.parametrize("name", NAMES)
def test_document_and_sweep_match_camber(name):
    ref = GOLDEN["documents"][name]
    hull = load(name)
    # Decoding (including camber's version-1 conversion for the v1 document).
    assert onp.allclose(flat(hull.to_dict()), flat(ref["state"]), rtol=0, atol=1e-9 * hull.loa)
    sw = SweptHull(hull)
    tol = 1e-11 * hull.loa
    for r in ref["plan"]:
        assert onp.allclose(sw.plan.at(r["u"]), r["at"], atol=tol)
        assert onp.allclose(sw.plan.d(r["u"]), r["d"], atol=tol)
    for r in ref["planUAtX"]:
        assert abs(sw.plan.u_at_x(r["x"]) - r["u"]) < 1e-12
    for r in ref["trimZ"]:
        assert abs(sw.trim_z(r["x"]) - r["z"]) < tol
    for r in ref["loft"]:
        pts, ks, keel_k = sw.loft.at(r["u"])
        assert onp.allclose(pts, r["pts"], atol=tol)
        assert onp.allclose(ks, r["ks"], atol=1e-12)
        assert abs(keel_k - r["keelK"]) < 1e-12
    for r in ref["frames"]:
        p, T, n = sw.frame(r["u"])
        assert onp.allclose(p, r["p"], atol=tol) and onp.allclose(T, r["T"], atol=1e-12)
        assert onp.allclose(n, r["n"], atol=1e-12)
    for r in ref["sections"]:
        sec = sw.section(r["u"])
        assert onp.allclose(sec.at(onp.array(r["v"])), r["at"], atol=tol)
        assert onp.allclose(sec.d(onp.array(r["v"])), r["d"], atol=tol)
    for r in ref["swept"]:
        s = sw.swept_section(r["u"], 4)
        assert (s["pts"] is None) == r["empty"]
        if not r["empty"]:
            assert s["keel"] == r["keel"]
            assert onp.allclose(s["pts"], r["pts"], atol=tol)


def test_json_round_trip(tmp_path):
    for name in NAMES:
        hull = load(name)
        path = tmp_path / f"{name}.json"
        hull.to_json(str(path))
        again = CamberHull.from_json(str(path))
        assert onp.allclose(flat(again.to_dict()), flat(hull.to_dict()))
        assert again.unit == hull.unit and again.waterline == pytest.approx(hull.waterline)
        assert again.deck_trim == pytest.approx(hull.deck_trim)


def test_refuses_newer_versions():
    doc = load("skiff").to_dict()
    doc["version"] = 3
    with pytest.raises(ValueError, match="version"):
        parse_document(doc)


@pytest.mark.parametrize("family", range(len(GOLDEN["families"])))
def test_promote_and_blend_match_camber(family):
    ref = GOLDEN["families"][family]
    promoted = promote_family([load(n) for n in ref["names"]])
    first = promoted[0]
    for mine, theirs in zip(promoted, ref["promoted"]):
        assert onp.allclose(flat(mine.to_dict()), flat(theirs), atol=1e-9 * first.loa)
    for b in ref["blends"]:
        mixed = blend(promoted, b["weights"])
        assert onp.allclose(flat(mixed.to_dict()), flat(b["state"]), atol=1e-9 * first.loa)
        sw = SweptHull(mixed)
        for r in b["swept"]:
            s = sw.swept_section(r["u"], 4)
            assert (s["pts"] is None) == r["empty"]
            if not r["empty"]:
                assert onp.allclose(s["pts"], r["pts"], atol=1e-10 * first.loa)


@pytest.mark.parametrize("name", NAMES)
def test_hull_hydrostatics_match_camber(name):
    """The offset table's displacement and LCB against camber's own hydrostatics."""
    ref = GOLDEN["documents"][name]["hydroFine"]
    doc = load(name)
    hull = Hull.from_camber(doc)
    statics = hull.hydrostatics()
    um = doc.unit_m
    assert float(statics["volume"]) == pytest.approx(ref["vol"] * um**3, rel=0.01)
    # camber measures x forward from the plan's start; the hull's x is aft.
    assert -float(statics["x_cb"]) == pytest.approx(ref["lcb"] * um, abs=0.005 * doc.loa * um)


def test_level_oversampling_conserves_area():
    """Band-averaged levels keep the volume, and move smoothly with the geometry."""
    doc = load("skiff")
    plain = Hull.from_camber(doc, level_oversampling=1).hydrostatics()["volume"]
    averaged = Hull.from_camber(doc, level_oversampling=8).hydrostatics()["volume"]
    assert float(averaged) == pytest.approx(float(plain), rel=0.01)


def test_lattice_basis_interpolates():
    for n_hulls, degree in ((2, 4), (3, 3), (4, 2)):
        lattice = simplex_lattice(n_hulls, degree)
        for i, a in enumerate(lattice):
            basis = lattice_basis([a_i / degree for a_i in a], lattice, degree)
            assert onp.allclose(basis, onp.eye(len(lattice))[i], atol=1e-12)
        # A partition of unity everywhere on the simplex.
        w = onp.random.default_rng(0).dirichlet(onp.ones(n_hulls))
        assert sum(lattice_basis(w, lattice, degree)) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def family():
    return HullFamily([HERE / "documents" / "skiff.json", HERE / "documents" / "dory.json"])


def test_family_interpolant(family):
    assert len(family.nodes) == 5
    # Exact at a node, close in between.
    node = family.at([0.5, 0.5]).hydrostatics()
    assert float(node["volume"]) == pytest.approx(float(family.nodes[2].hydrostatics()["volume"]), rel=1e-12)
    for t in (0.1, 0.37, 0.8):
        approx = family.at([1 - t, t]).hydrostatics()
        exact = family.exact([1 - t, t]).hydrostatics()
        assert float(approx["volume"]) == pytest.approx(float(exact["volume"]), rel=0.005)
        assert float(approx["x_cb"]) == pytest.approx(float(exact["x_cb"]), abs=0.01)


def test_family_blend_is_differentiable(family):
    opti = asb.Opti()
    t = opti.variable(init_guess=0.3, lower_bound=0.0, upper_bound=1.0)
    hull = family.at([1 - t, t])
    volume = hull.hydrostatics(0.0, 0.0, (-2.4, 0.0))["volume"]
    # Find the blend with a given displacement: a smooth 1-D root.
    target = float(family.exact([0.6, 0.4]).hydrostatics()["volume"])
    opti.minimize((volume / target - 1.0) ** 2)
    sol = opti.solve(verbose=False)
    assert sol(t) == pytest.approx(0.4, abs=0.02)


def test_curve_promotion_keeps_the_shape():
    """Adding section points on the curve leaves the hull nearly unchanged."""
    skiff, dory = load("skiff"), load("dory")  # 5 and 4 points per section
    promoted = promote_family([dory, skiff], section_points="curve")[0]
    assert promoted.topology[2] == 5
    before = float(Hull.from_camber(dory).hydrostatics()["volume"])
    after = float(Hull.from_camber(promoted).hydrostatics()["volume"])
    assert after == pytest.approx(before, rel=0.005)


def test_scaled_and_family_length():
    skiff = load("skiff")
    half = skiff.scaled(0.5)
    assert half.loa == pytest.approx(0.5 * skiff.loa)
    assert half.waterline == pytest.approx(0.5 * skiff.waterline)
    v_full = float(Hull.from_camber(skiff).hydrostatics()["volume"])
    v_half = float(Hull.from_camber(half).hydrostatics()["volume"])
    assert v_half == pytest.approx(v_full / 8, rel=0.01)


def test_beam_factor_squeezes_breadth():
    skiff = load("skiff")
    narrow = skiff.scaled(1.0, 0.5)
    assert narrow.loa == pytest.approx(skiff.loa)
    v_full = float(Hull.from_camber(skiff).hydrostatics()["volume"])
    v_narrow = float(Hull.from_camber(narrow).hydrostatics()["volume"])
    assert v_narrow == pytest.approx(0.5 * v_full, rel=0.03)


def test_surface_patches_follow_the_sweep():
    """CAD patches reproduce the swept surface between the samples they interpolate."""
    from scipy.interpolate import BSpline

    sw = SweptHull(load("skiff"))
    patches = sw.surface_patches(n_columns=61, both_sides=False)
    assert patches[-1]["degree"] == (1, 1)  # the transom
    breaks = sw.smooth_intervals()
    span, S, R = breaks[-1] - breaks[0], sw.loft.S, 8
    k, worst = 0, 0.0
    for u_a, u_b in zip(breaks[:-1], breaks[1:]):
        n = max(6, int(round(61 * (u_b - u_a) / span)))
        sf = onp.linspace(0, 1, 2 * n - 1)
        us = (u_a + 1e-9 * span) + (u_b - u_a - 2e-9 * span) * 0.5 * (1 - onp.cos(onp.pi * sf))
        fine = sw.to_hull_axes(onp.stack([sw.swept_section(float(u), 2 * R)["pts"] for u in us]))
        ks = onp.array([sw.loft.at(float(u))[1] for u in us[::2]])
        cuts = [0] + [i * R for i in range(1, S - 1) if ks[:, i].max() > 0.05] + [(S - 1) * R]
        for a, b in zip(cuts[:-1], cuts[1:]):
            p = patches[k]
            k += 1
            along = BSpline(onp.array(p["knots_u"]), p["ctrl"], p["degree"][0], axis=0)(sf)
            surf = BSpline(onp.array(p["knots_v"]), along, p["degree"][1], axis=1)(onp.linspace(0, 1, 2 * (b - a) + 1))
            worst = max(worst, float(onp.linalg.norm(surf - fine[:, 2 * a : 2 * b + 1], axis=-1).max()))
    assert k == len(patches) - 1
    assert worst < 0.002 * load("skiff").loa * load("skiff").unit_m


def test_generated_demihull():
    """A hull from numbers alone: valid, few points, square-sterned, at the asked displacement."""
    from hydrosandbox.hull.camber.generate import DemihullParameters, demihull, with_displacement

    for p in (
        DemihullParameters(),
        DemihullParameters(station_u=(0.0, 0.3, 0.7, 1.0), side=(300.0, 380.0, 450.0, 480.0),
                           bottom=(150.0, 220.0, 230.0, 200.0), fullness=(0.9, 0.6, 0.4, 0.2)),
    ):
        doc = with_displacement(demihull(p), 0.25, n_stations=41, n_levels=25)
        assert doc.topology == (5, 3, 4, p.n_stations)
        again = parse_document(json.loads(doc.to_json()))  # a valid version-2 document
        hull = Hull.from_camber(again, n_stations=41, n_levels=25)
        assert float(hull.hydrostatics()["volume"]) == pytest.approx(0.25, rel=2e-3)
        assert hull.length == pytest.approx(p.length / 1000.0, rel=1e-3)
        assert hull.has_transom
        # Narrower stations are cut nearer the sheer: the keel rises to the stem.
        assert float(hull.keel_height(hull.stations[1])) > float(hull.keel_height(hull.stations[20]))
        # The bow closes: nothing at the stem swings outboard of the centreline.
        stem = SweptHull(doc).swept_section(1.0, 8)["pts"]
        assert stem is None or onp.abs(stem[:, 1]).max() < 0.5


def test_generated_bow_closes_with_a_smooth_foot():
    """A section whose smooth topsides would bulge outboard gets just enough knuckle."""
    from hydrosandbox.hull.camber.generate import inboard, outboard_bulge, section_points

    pts = section_points(side=500.0, bottom=100.0, fullness=0.2, flare=38.0, reach=298.0)
    assert outboard_bulge(pts) > 5.0
    fixed = inboard(pts)
    assert outboard_bulge(fixed) <= 0.05 and 0.0 < fixed[1].k <= 1.0
