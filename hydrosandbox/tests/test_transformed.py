import numpy as onp
import pytest

import aerosandbox as asb
from hydrosandbox.hull.hull import Hull
from hydrosandbox.hull.transformed import (
    TransformedFamily,
    area_curve_coefficients,
    box_design,
    lackenby,
    transform_hull,
)


@pytest.fixture(scope="module")
def wigley():
    return Hull.wigley(length=5.5, beam=0.4, draft=0.25)


def test_neutral_transform_is_identity(wigley):
    t = transform_hull(wigley)
    assert onp.allclose(t.stations, wigley.stations)
    assert onp.allclose(t.levels, wigley.levels)
    assert onp.allclose(t.half_breadths, wigley.half_breadths)


def test_beam_and_draft_scale_volume(wigley):
    v0 = float(wigley.hydrostatics()["volume"])
    assert float(transform_hull(wigley, beam=1.1).hydrostatics()["volume"]) == pytest.approx(1.1 * v0, rel=1e-9)
    assert float(transform_hull(wigley, draft=0.9).hydrostatics()["volume"]) == pytest.approx(0.9 * v0, rel=1e-4)
    t = transform_hull(wigley, draft=0.9)
    assert onp.allclose(t.levels[t.levels >= 0], wigley.levels[wigley.levels >= 0])  # topsides kept


def test_lackenby_reaches_targets_with_ends_fixed(wigley):
    base = area_curve_coefficients(wigley.stations, wigley.levels, wigley.half_breadths)
    x = lackenby(wigley.stations, wigley.levels, wigley.half_breadths, cp=0.03, lcb=-0.02)
    c = area_curve_coefficients(x, wigley.levels, wigley.half_breadths)
    assert c["cp"] == pytest.approx(base["cp"] + 0.03, abs=1e-7)
    assert c["lcb"] == pytest.approx(base["lcb"] - 0.02, abs=1e-7)
    assert x[0] == wigley.stations[0] and x[-1] == wigley.stations[-1]
    assert onp.all(onp.diff(x) > 0)
    with pytest.raises(ValueError):
        lackenby(wigley.stations, wigley.levels, wigley.half_breadths, cp=0.3)


def test_bow_rake_keeps_waterline_and_afterbody(wigley):
    t = transform_hull(wigley, bow_rake=0.5)
    assert t.stations[0] < wigley.stations[0]  # stem raked forward above water
    j0 = int(onp.argmin(abs(wigley.levels)))
    x_wl = lambda h: h.stations[onp.argmax(h.half_breadths[:, j0] > 1e-9)]
    assert x_wl(t) == pytest.approx(x_wl(wigley), abs=0.05)
    aft = wigley.stations > 0.5
    f_aft = onp.array([onp.interp(wigley.stations[aft], t.stations, t.half_breadths[:, j]) for j in range(len(t.levels))]).T
    assert onp.allclose(f_aft, wigley.half_breadths[aft], atol=1e-9)


def test_box_design_is_unisolvent():
    for n in (1, 2, 4):
        for degree in (1, 2):
            nodes, monomials = box_design(n, degree)
            assert len(nodes) == len(monomials) == (n + 1 if degree == 1 else (n + 1) * (n + 2) // 2)
            v = onp.array([[onp.prod([u ** e for u, e in zip(node, m)]) for m in monomials] for node in nodes])
            assert abs(onp.linalg.det(v)) > 1e-9


@pytest.fixture(scope="module")
def family(wigley):
    return TransformedFamily(
        wigley, {"beam": (0.9, 1.1), "draft": (0.85, 1.15), "cp": (-0.03, 0.03), "bow_rake": (0.0, 0.5)}
    )


def test_family_basis_and_interpolation(family):
    for i, p in enumerate(family.node_parameters):
        assert onp.allclose(family.basis(p), onp.eye(len(family.nodes))[i], atol=1e-12)
    v = {"beam": 1.05, "draft": 0.9, "cp": 0.01, "bow_rake": 0.3}
    assert sum(family.basis(v)) == pytest.approx(1.0)
    a, e = family.at(v).hydrostatics(), family.exact(v).hydrostatics()
    for k in ("volume", "x_cb", "z_cb", "wetted_surface"):
        assert float(a[k]) == pytest.approx(float(e[k]), rel=3e-3, abs=1e-4)


def test_family_is_differentiable(family):
    opti = asb.Opti()
    beam = opti.variable(init_guess=1.0, lower_bound=0.9, upper_bound=1.1)
    volume = family.at({"beam": beam, "draft": 1.0, "cp": 0.0, "bow_rake": 0.0}).hydrostatics()["volume"]
    target = 1.05 * float(family.base.hydrostatics()["volume"])
    opti.subject_to(volume == target)
    sol = opti.solve(verbose=False)
    assert sol(beam) == pytest.approx(1.05, rel=2e-3)
