import numpy as onp
import pytest

from hydrosandbox.hull.hull import Hull
from hydrosandbox.hull.ffd import FFDFamily, FFDLattice


@pytest.fixture(scope="module")
def wigley():
    return Hull.wigley(length=5.5, beam=0.4, draft=0.25)


@pytest.fixture(scope="module")
def lattice(wigley):
    return FFDLattice(wigley, nx=4, nz=3)


def test_no_moves_reproduce_the_hull(wigley, lattice):
    h = lattice.hull({})
    assert onp.allclose(h.stations, wigley.stations, atol=1e-12)
    assert onp.allclose(h.levels, wigley.levels, atol=1e-12)
    assert onp.allclose(h.half_breadths, wigley.half_breadths, atol=1e-12)


def test_names_and_validation(lattice):
    names = lattice.names()
    # Per frame: centreplane x, z and outboard x, y, z at each of nz layers.
    assert len(names) == 4 * 3 * 5
    assert "p0i0y" not in names and "p0o0y" in names
    with pytest.raises(ValueError):
        lattice.moves({"p1i1y": 0.01})
    with pytest.raises(ValueError):
        lattice.moves({"p9o0x": 0.01})


def test_moves_are_linear_in_the_surface(lattice):
    a = {"p1o1y": 0.02, "p2i0z": -0.03}
    b = {"p1o1y": 0.04, "p2i0z": -0.06}
    base = lattice.deform({})
    one, two = lattice.deform(a), lattice.deform(b)
    for k in range(3):
        assert onp.allclose(two[k] - base[k], 2.0 * (one[k] - base[k]), atol=1e-12)


def test_sideways_moves_are_exact_in_a_linear_family(lattice):
    names = [f"p{i}o{k}y" for i in range(4) for k in range(3)]
    fam = FFDFamily(lattice, names, half_width=0.03, design="linear")
    assert len(fam.nodes) == len(names) + 1
    v = dict(zip(names, onp.random.default_rng(0).uniform(-0.03, 0.03, len(names))))
    approx, exact = fam.at(v).hydrostatics(), fam.exact(v).hydrostatics()
    # Volume is linear in the half-breadths, so exact; the centroids are ratios.
    assert float(approx["volume"]) == pytest.approx(float(exact["volume"]), rel=1e-9)
    for k in ("x_cb", "z_cb"):
        assert float(approx[k]) == pytest.approx(float(exact[k]), abs=1e-3)


def test_folding_is_refused(lattice):
    with pytest.raises(ValueError):
        lattice.hull({"p1i0z": 2.0, "p1o0z": 2.0})


def test_recentred_family_moves_its_centre(lattice):
    fam = FFDFamily(lattice, ["p1o1y", "p2o1y"], half_width=0.02)
    step = fam.recentred({"p1o1y": 0.015, "p2o1y": -0.01})
    assert step.start == {"p1o1y": 0.015, "p2o1y": -0.01}
    assert step.ranges["p1o1y"] == pytest.approx((-0.005, 0.035))
    assert onp.allclose(step.exact({"p1o1y": 0.015, "p2o1y": -0.01}).half_breadths,
                        lattice.hull({"p1o1y": 0.015, "p2o1y": -0.01}).half_breadths)


def test_no_fold_constraints_keep_solutions_buildable(lattice):
    import aerosandbox as asb

    names = lattice.names()
    fam = FFDFamily(lattice, names, 0.05)
    assert all(fam.no_fold({}))
    bad = {f"p{i}i0z": 0.3 for i in range(4)} | {f"p{i}i1z": -0.3 for i in range(4)}
    assert not all(fam.no_fold(bad))
    with pytest.raises(ValueError):
        lattice.hull(bad)
    # A solve driven toward folding stops where the hull can still be built.
    opti = asb.Opti()
    v = [opti.variable(init_guess=0.0, lower_bound=-0.3, upper_bound=0.3) for _ in names]
    opti.subject_to(fam.no_fold(v))
    opti.minimize(sum(v[names.index(f"p{i}i1z")] - v[names.index(f"p{i}i0z")] for i in range(4)))
    sol = opti.solve(verbose=False)
    lattice.hull({n: float(sol(x)) for n, x in zip(names, v)})
