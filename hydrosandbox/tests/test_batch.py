import numpy as onp
import pytest

from hydrosandbox.hull.hull import Hull, dynamic_force_batch, wave_amplitude_batch


@pytest.fixture(scope="module")
def hulls():
    base = Hull.wigley(length=5.5, beam=0.4, draft=0.25)
    # Same offset grid, different half-breadths (a family's nodes).
    return [Hull(base.stations, base.levels, base.half_breadths * f) for f in (0.9, 1.0, 1.15)]


@pytest.mark.parametrize("sinkage, trim", [(0.0, 0.0), (-0.03, 1.5)])
def test_dynamic_force_batch_matches_each_hull(hulls, sinkage, trim):
    args = (6.0, 1025.0, 9.80665)
    batch = dynamic_force_batch(hulls, *args, sinkage, trim, (0.0, 0.0), demihull_spacing=2.2)
    for j, h in enumerate(hulls):
        one = h.dynamic_force(*args, sinkage=sinkage, trim=trim, demihull_spacing=2.2)
        for k, v in batch.items():
            if k in one:
                assert onp.ravel(v)[j] == pytest.approx(float(onp.ravel(one[k])[0]), rel=1e-9, abs=1e-9), k


def test_wave_amplitude_batch_matches_each_hull(hulls):
    kx = onp.linspace(0.5, 3.0, 7)
    ky = onp.linspace(0.0, 2.0, 7)
    kappa = onp.sqrt(kx**2 + ky**2)
    re, im = wave_amplitude_batch(hulls, 6.0, 9.80665, -0.02, 1.0, (0.0, 0.0), 2.2, kx, kappa, ky)
    for j, h in enumerate(hulls):
        r1, i1 = h.wave_amplitude(6.0, 9.80665, sinkage=-0.02, trim=1.0, demihull_spacing=2.2)(kx, kappa, ky)
        assert onp.allclose(re[j], onp.ravel(r1), rtol=1e-9, atol=1e-12)
        assert onp.allclose(im[j], onp.ravel(i1), rtol=1e-9, atol=1e-12)
