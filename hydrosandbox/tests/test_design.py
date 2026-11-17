"""The design API: loads and scenarios, the objective, and optimize / evaluate."""

import numpy as onp
import pytest

import hydrosandbox.design as hd
from hydrosandbox.design.model import centre_of_gravity
from hydrosandbox.hull import Hull


@pytest.fixture(scope="module")
def hull():
    return Hull.wigley(length=5.5, beam=0.42, draft=0.25)


def test_distributions():
    assert hd.Normal(80, 15).mean() == 80 and hd.Normal(80, 15).quantile(0.5) == pytest.approx(80)
    assert hd.Normal(0, 1).quantile(0.975) == pytest.approx(1.96, abs=0.01)
    assert hd.Uniform(1, 3).mean() == 2 and hd.Uniform(1, 3).quantile(0.25) == pytest.approx(1.5)
    d = hd.Discrete([10, 20, 40], [1, 1, 2])
    assert d.mean() == pytest.approx(27.5)
    assert [d.quantile(u) for u in (0.1, 0.3, 0.6, 0.99)] == [10, 20, 40, 40]
    assert hd.Free(1, 3).initial() == 2 and hd.Free(1, 3, start=1.5).initial() == 1.5


def test_load_rules(hull):
    with pytest.raises(ValueError, match="cannot be Free"):
        hd.PointMass("a", hd.Free(1, 2), [0, 0, 0])
    with pytest.raises(ValueError, match="cannot also have an uncertain"):
        hd.PointMass("a", 10.0, [hd.Free(0, 1), 0, hd.Normal(0, 0.1)])
    with pytest.raises(ValueError, match="distinct names"):
        hd.Boat(hull, 100.0, [0, 0, 0], loads=[hd.PointMass("a", 1.0, [0, 0, 0]), hd.PointMass("a", 1.0, [0, 0, 0])])


def test_scenarios_reproduce_the_spread_of_mass_and_cg(hull):
    # No uncertainty: one scenario, the boat as given.
    scenarios, notes = hd.load_scenarios(hd.Boat(hull, 520.0, [0.3, 0, 0.1]))
    assert len(scenarios) == 1 and scenarios[0].weight == 1 and scenarios[0].mass == 520 and not notes
    assert scenarios[0].moment == pytest.approx([156.0, 0.0, 52.0])

    # One uncertain mass at a fixed point: one direction of spread, three points, 2/3 and 1/6 each.
    boat = hd.Boat(hull, 440.0, [0.3, 0, 0], loads=[hd.PointMass("crew", hd.Normal(80, 50), [0.3, 0, 0])])
    scenarios, _ = hd.load_scenarios(boat)
    weights = onp.array([s.weight for s in scenarios])
    masses = onp.array([s.mass for s in scenarios])
    assert weights == pytest.approx([2 / 3, 1 / 6, 1 / 6])
    assert sorted(masses) == pytest.approx([520 - 50 * 3**0.5, 520, 520 + 50 * 3**0.5], abs=0.5)

    # Mass and position both uncertain: mean and variance of the mass and of the moment are kept.
    boat = hd.Boat(hull, 400.0, [0.3, 0, 0], loads=[hd.PointMass("crew", hd.Normal(80, 15), [hd.Normal(0.5, 0.3), 0, 0.5])])
    scenarios, _ = hd.load_scenarios(boat)
    w = onp.array([s.weight for s in scenarios])
    assert w.sum() == pytest.approx(1) and len(scenarios) == 5
    mass = onp.array([s.mass for s in scenarios])
    moment = onp.array([s.moment[0] for s in scenarios])
    assert w @ mass == pytest.approx(480, abs=0.05) and onp.sqrt(w @ (mass - w @ mass) ** 2) == pytest.approx(15, rel=0.01)
    expected_sd = onp.sqrt((15 * 0.5) ** 2 + (80 * 0.3) ** 2 + (15 * 0.3) ** 2)  # of m * x, independent
    assert w @ moment == pytest.approx(400 * 0.3 + 80 * 0.5, abs=0.1)
    assert onp.sqrt(w @ (moment - w @ moment) ** 2) == pytest.approx(expected_sd, rel=0.01)


def test_optional_and_placed_loads_and_explicit_cases(hull):
    boat = hd.Boat(hull, 400.0, [0.3, 0, 0], loads=[
        hd.PointMass("crew", hd.Normal(80, 15), [0.5, hd.Normal(0, 0.2), 0.5]),
        hd.PointMass("passenger", 75.0, [-0.7, 0, 0.5], probability=0.25),
        hd.PointMass("battery", 40.0, [hd.Free(-1, 1.5), 0, 0]),
    ])
    scenarios, notes = hd.load_scenarios(boat)
    aboard = [s for s in scenarios if "passenger aboard" in s.name]
    assert sum(s.weight for s in aboard) == pytest.approx(0.25)
    assert sum(s.weight for s in scenarios) == pytest.approx(1)
    assert all(s.placed_masses == {"battery": 40.0} for s in scenarios)
    assert any("sideways" in note for note in notes)  # the crew's sideways spread has no effect yet
    # The placed load's position is a design variable; the CG follows it.
    mass, cg = centre_of_gravity(boat, {"battery.x": 1.0}, 500.0, [100.0, 0.0, 50.0], {"battery": 40.0})
    assert mass == 540 and cg[0] == pytest.approx(140 / 540) and cg[2] == pytest.approx(50 / 540)

    # The fixed part's own position can be the optimizer's, within a small range.
    shiftable = hd.Boat(hull, 400.0, [hd.Free(0.1, 0.5), 0, 0.2], loads=[hd.PointMass("crew", hd.Normal(80, 15), [0.5, 0, 0.5])])
    assert shiftable.placed_names() == ["fixed"]
    for s in hd.load_scenarios(shiftable)[0]:
        assert s.placed_masses == {"fixed": 400.0} and s.moment[0] == pytest.approx(s.mass * 0.5)
    mass, cg = centre_of_gravity(shiftable, {"fixed.x": 0.4}, 80.0, [40.0, 0.0, 40.0], {"fixed": 400.0})
    assert mass == 480 and cg[0] == pytest.approx((40 + 160) / 480) and cg[2] == pytest.approx((40 + 80) / 480)
    with pytest.raises(ValueError, match="cannot be uncertain"):
        hd.Boat(hull, 400.0, [hd.Normal(0.3, 0.1), 0, 0])
    with pytest.raises(ValueError, match="reserved"):
        hd.Boat(hull, 400.0, [0, 0, 0], loads=[hd.PointMass("fixed", 1.0, [0, 0, 0])])

    nominal, _ = hd.load_scenarios(boat, 1)
    assert len(nominal) == 1 and nominal[0].mass == pytest.approx(400 + 80 + 0.25 * 75)
    drawn, _ = hd.load_scenarios(boat, 16)
    assert len(drawn) == 16 and sum(s.weight for s in drawn) == pytest.approx(1)
    cases, _ = hd.load_scenarios(boat, [
        {"name": "light", "weight": 1, "loads": {"passenger": {"aboard": False}, "crew": {"mass": 60}}},
        {"name": "heavy", "weight": 3, "loads": {"crew": {"mass": 110, "position": [0.2, 0, 0.5]}}},
    ])
    assert [s.name for s in cases] == ["light", "heavy"] and [s.weight for s in cases] == pytest.approx([0.25, 0.75])
    assert cases[0].mass == 460 and cases[1].mass == 585
    with pytest.raises(KeyError):
        hd.load_scenarios(boat, [{"loads": {"nobody": {}}}])


def test_objective_terms():
    q = {"damping": 20.0, "drag": 440.0, "stress": 165e6, "margin": 0.02}
    plain = hd.Objective(maximize="damping")
    assert float(plain.utility(q, {})) == pytest.approx(20.0, rel=1e-6)
    # A cost: (reference / q) ^ exponent.
    cost = hd.Objective(maximize="damping", costs=[hd.Cost("drag", exponent=8.0)])
    assert float(cost.utility(q, {0: 400.0})) == pytest.approx(20.0 * (400 / 440) ** 8, rel=1e-6)
    # Costs alone: minimizing drag.
    assert float(hd.Objective(costs=[hd.Cost("drag")]).utility(q, {0: 400.0})) == pytest.approx(400 / 440)
    # A penalty: 10 percent past the threshold is one unit, and costs exp(-weight).
    over = hd.Objective(maximize="damping", penalties=[hd.AtMost("stress", 150e6, weight=2.0)])
    assert float(over.utility(q, {})) == pytest.approx(20.0 * onp.exp(-2.0), rel=1e-6)
    inside = hd.Objective(maximize="damping", penalties=[hd.AtMost("stress", 200e6), hd.AtLeast("margin", 0.0, scale=0.05)])
    assert float(inside.utility(q, {})) == pytest.approx(20.0, rel=1e-6) and inside.violations(q) == []
    under = hd.Objective(maximize="damping", penalties=[hd.AtLeast("margin", 0.1, scale=0.04)])
    assert float(under.utility(q, {})) == pytest.approx(20.0 * onp.exp(-4.0), rel=1e-6)
    report = under.violations(q)
    assert report[0]["quantity"] == "margin" and report[0]["violation"] == pytest.approx(2.0)
    # Zero exponent or weight switches a term off; the logarithm agrees with the utility.
    off = hd.Objective(maximize="damping", costs=[hd.Cost("drag", exponent=0.0)], penalties=[hd.AtMost("stress", 150e6, weight=0.0)])
    assert float(off.utility(q, {0: 400.0})) == pytest.approx(20.0, rel=1e-6)
    assert float(over.log_utility(q, {})) == pytest.approx(onp.log(20.0) - 2.0, rel=1e-6)
    # Far past a threshold the utility underflows but its logarithm does not.
    far = dict(q, stress=1500e6)
    assert float(over.utility(far, {})) == 0.0 and onp.isfinite(float(over.log_utility(far, {})))
    # A custom quantity, and what an objective needs.
    custom = hd.Objective(maximize=lambda named: named["damping"] / named["drag"])
    assert float(custom.utility(q, {}, measure_scale=20 / 440)) == pytest.approx(20 / 440, rel=1e-6)
    assert custom.needed() is None
    assert cost.needed() == {"damping", "drag"}
    with pytest.raises(ValueError, match="scale"):
        hd.AtLeast("margin", 0.0)
    with pytest.raises(KeyError, match="Unknown quantity"):
        hd.Objective(maximize="nothing").utility(q, {})


def test_evaluate_and_optimize_on_a_catamaran(hull):
    """A design is assessed in every scenario, and a short optimization improves the objective."""
    x0 = float(hull.stations[0])
    boat = hd.Boat(
        hull, fixed_mass=440.0, fixed_cg=[x0 + 2.9, 0.0, 0.0], demihull_spacing=2.2,
        loads=[hd.PointMass("crew", hd.Normal(80, 30), [x0 + 2.9, 0.0, 0.0])],
        craft_options=dict(wave_length_scales=(3.0, 8.0, 0.01)),
    )
    design = hd.two_strut_foil(boat, velocity=6.0)
    design.start.update(root_chord=0.25)
    objective = hd.Objective(
        maximize="pitch_damping", costs=[hd.Cost("drag", exponent=4.0)],
        penalties=hd.standard_penalties(design) + [hd.AtMost("bending_stress", 400e6)],
    )
    assessed = hd.evaluate(boat, 6.0, design, objective=objective)
    assert len(assessed["scenarios"]) == 3
    masses = [s["mass"] for s in assessed["scenarios"]]
    assert masses[0] == pytest.approx(520, abs=0.1) and max(masses) - min(masses) == pytest.approx(2 * 30 * 3**0.5, rel=0.02)
    for s in assessed["scenarios"]:
        assert set(hd.MODEL_QUANTITIES) <= set(s["quantities"]) and "bending_stress" in s["quantities"]
        assert s["quantities"]["pitch_damping"] > 0 and s["quantities"]["drag"] > 0
    # Heavier boat, more drag.
    by_mass = sorted(assessed["scenarios"], key=lambda s: s["mass"])
    assert by_mass[0]["quantities"]["drag"] < by_mass[-1]["quantities"]["drag"]

    # Against a fixed reference a lighter boat scores better merely for being lighter;
    # against the baseline in the same scenario most of that goes.
    def spread(result):
        utilities = [s["utility"] for s in result["scenarios"]]
        return max(utilities) / min(utilities)

    objective.costs[0].reference = "baseline"
    relative = hd.evaluate(boat, 6.0, design, objective=objective)
    assert len(relative["baseline"]) == 3 and all(b["pitch_damping"] == 0 for b in relative["baseline"])
    for s, b, r in zip(relative["scenarios"], relative["baseline"], relative["references"]):
        assert r[0] == pytest.approx(b["drag"]) and b["mass"] == pytest.approx(s["mass"])
    assert spread(assessed) > 1.5 and spread(relative) - 1 < 0.25 * (spread(assessed) - 1)

    out = hd.optimize(boat, 6.0, design, objective, solver=hd.Solver(max_iterations=4, table_jobs=4))
    assert len(out["baseline"]) == 3
    assert out["status"] in ("converged", "iteration limit") and out["iterations"] <= 4
    assert set(out["design"]) == set(design.variables)
    assert all(design.variables[n][0] - 1e-6 <= v <= design.variables[n][1] + 1e-6 for n, v in out["design"].items())
    assert len(out["scenarios"]) == 3 and out["expected"]["utility"]["mean"] > 0
