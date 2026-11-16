"""``optimize()`` and ``evaluate()``: design a boat for its loads, or assess a given design."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import casadi as cas
import numpy as onp

from hydrosandbox.design.boat import Boat
from hydrosandbox.design.design_space import DesignSpace
from hydrosandbox.design.model import (
    ThrustAt,
    as_numbers,
    balance,
    centre_of_gravity,
    hull_tables,
    placed_variables,
    quantities,
)
from hydrosandbox.design.objective import Objective
from hydrosandbox.design.scenarios import Scenario, load_scenarios


@dataclass
class Solver:
    """
    How the problem is solved.

    ``trim_bounds`` [deg] and ``sinkage_bounds`` [m] bound each scenario's attitude.
    ``threads`` evaluates scenarios in parallel (memory grows with it, not with the
    number of scenarios). ``table_jobs`` is the number of processes used to build
    the hull's attitude tables. ``max_seconds`` stops the solver after that much
    time, returning the design it has reached. ``inline`` builds the model anew for
    every scenario instead of once as a function (memory then grows with the number
    of scenarios; for comparison and for single-scenario problems). ``progress`` prints one line per
    iteration (the objective and the largest balance residual); ``verbose`` prints
    the solver's own output.
    """

    max_iterations: int = 200
    trim_bounds: tuple[float, float] = (-3.0, 6.0)
    sinkage_bounds: tuple[float, float] = (-0.4, 0.2)
    threads: int = 1
    table_jobs: int | None = 1
    max_seconds: float | None = None
    inline: bool = False
    progress: bool = False
    verbose: bool = False


_STATUS = {
    "Solve_Succeeded": "converged",
    "Solved_To_Acceptable_Level": "converged",
    "Maximum_Iterations_Exceeded": "iteration limit",
    "Infeasible_Problem_Detected": "infeasible",
    "Maximum_WallTime_Exceeded": "time limit",
}


def _all_variables(boat: Boat, design: DesignSpace):
    """Names, bounds and starts of every design variable: the design space's, then placed loads'."""
    names = list(design.variables)
    low = [design.variables[n][0] for n in names]
    high = [design.variables[n][1] for n in names]
    start = [design.start[n] for n in names]
    for name, (lo, hi, initial) in placed_variables(boat).items():
        if name in design.variables:
            raise ValueError(f"A placed load and a design variable share the name {name!r}.")
        names.append(name)
        low.append(lo)
        high.append(hi)
        start.append(initial)
    return names, low, high, start


def _references(objective: Objective, nominal: dict, baseline: list[dict] | None, n: int) -> list[dict[int, float]]:
    """Each cost's reference in each scenario: its own number, the baseline's value there, or the nominal value."""
    per_scenario = []
    for k in range(n):
        refs = {}
        for i, cost in enumerate(objective.costs):
            if cost.reference == "baseline":
                refs[i] = float(_cost_value(cost, baseline[k]))
            elif cost.reference is None:
                refs[i] = float(_cost_value(cost, nominal))
            else:
                refs[i] = float(cost.reference)
        per_scenario.append(refs)
    return per_scenario


def _needs_baseline(objective: Objective | None) -> bool:
    return objective is not None and any(c.reference == "baseline" for c in objective.costs)


def _summarize(objective: Objective, scenarios: list[Scenario], per_scenario: list[dict], references, scale):
    weights = onp.array([s.weight for s in scenarios])
    logs = onp.array([float(objective.log_utility(q, r, scale)) for q, r in zip(per_scenario, references)])
    utilities = onp.exp(logs)
    mean = float(weights @ utilities)
    sd = float(onp.sqrt(weights @ (utilities - mean) ** 2))
    expected = {"utility": {"mean": mean, "sd": sd, "log_mean": float(weights @ logs)}}
    for name in per_scenario[0]:
        column = onp.array([q[name] for q in per_scenario])
        m = float(weights @ column)
        expected[name] = {"mean": m, "sd": float(onp.sqrt(weights @ (column - m) ** 2))}
    return utilities, expected, mean - objective.risk_aversion * sd


def evaluate(
    boat: Boat,
    velocity: float,
    design: DesignSpace,
    values: dict[str, float] | None = None,
    objective: Objective | None = None,
    propulsion: ThrustAt | None = None,
    scenarios: Any = None,
    baseline: DesignSpace | None = None,
    references: list[dict[int, float]] | None = None,
    measure_scale: float | None = None,
    guesses: list[tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """
    Assess a given design: balance it in every scenario, with the hull integrated
    directly, and report every quantity, the utility and any violations.

    ``values`` are the design variables (the design space's start if None). The
    scenarios, the ``baseline`` and the meaning of ``objective`` are as for
    ``optimize()``. Cost references that are left open default to the nominal
    (first) scenario's own values.
    """
    names, _, _, start = _all_variables(boat, design)
    values = dict(zip(names, start)) if values is None else {n: float(values[n]) for n in names}
    scenario_list, notes = load_scenarios(boat, scenarios)
    placed_names = [load.name for load in boat.loads if load.placed]
    per_scenario, records = [], []
    for k, s in enumerate(scenario_list):
        mass, cg = centre_of_gravity(boat, values, s.mass, s.moment, s.placed_masses)
        guess = (0.0, 0.0) if guesses is None else guesses[k]
        try:
            sinkage, trim = balance(boat, velocity, design, values, mass, cg, propulsion, guess=guess)
        except RuntimeError as error:
            raise RuntimeError(f"Scenario {s.name!r} has no balance for this design: {error}") from error
        q, _, _ = quantities(boat, velocity, design, values, mass, cg, sinkage, trim, propulsion)
        per_scenario.append(as_numbers(q))
        records.append({"name": s.name, "weight": s.weight, "mass": float(mass), "cg": [float(c) for c in cg]})
    result = {"design": values, "notes": notes}
    if objective is not None:
        if references is None:
            base = None
            if _needs_baseline(objective):
                base = baseline_quantities(boat, velocity, baseline, propulsion, scenarios)
                result["baseline"] = base
            references = _references(objective, per_scenario[0], base, len(scenario_list))
        if measure_scale is None:
            measure_scale = _measure_scale(objective, per_scenario[0])
        utilities, expected, value = _summarize(objective, scenario_list, per_scenario, references, measure_scale)
        result.update(objective=value, expected=expected, references=references)
        for record, q, u in zip(records, per_scenario, utilities):
            record.update(utility=float(u), violations=objective.violations(q))
    for record, q in zip(records, per_scenario):
        record["quantities"] = q
    result["scenarios"] = records
    return result


def baseline_quantities(boat: Boat, velocity: float, baseline: DesignSpace | None = None,
                        propulsion: ThrustAt | None = None, scenarios: Any = None) -> list[dict[str, float]]:
    """
    The quantities of the baseline design in every scenario, with the hull integrated
    directly. The baseline is the boat with no appendages unless a design space is
    given, at that space's starting values; placed loads sit at their own starts.
    """
    from hydrosandbox.design.layouts import no_appendages

    assessed = evaluate(boat, velocity, baseline or no_appendages(), None, None, propulsion, scenarios)
    return [s["quantities"] for s in assessed["scenarios"]]


def _cost_value(cost, named):
    return cost.quantity(named) if callable(cost.quantity) else named[cost.quantity]


def _measure_scale(objective: Objective, named) -> float:
    if objective.maximize is None:
        return 1.0
    value = objective.maximize(named) if callable(objective.maximize) else named[objective.maximize]
    return max(abs(float(value)), 1e-9)


def optimize(
    boat: Boat,
    velocity: float,
    design: DesignSpace,
    objective: Objective,
    propulsion: ThrustAt | None = None,
    scenarios: Any = None,
    baseline: DesignSpace | None = None,
    solver: Solver | None = None,
) -> dict[str, Any]:
    """
    Find the design with the best expected utility over the boat's load scenarios.

    Parameters
    ----------
    boat : The hull, its fixed mass and its loads (``Boat``).
    velocity : Speed [m/s]; a number.
    design : The design variables and how they become geometry (``DesignSpace``).
    objective : What to maximize, the costs and the penalties (``Objective``).
    propulsion : None (thrust through the centre of gravity) or ``ThrustAt``.
    scenarios : How the loads' uncertainty becomes scenarios; see
        ``hydrosandbox.design.scenarios.load_scenarios``. None places a few points
        on the spread of the load summaries the model responds to.
    baseline : The design that costs with ``reference="baseline"`` are measured
        against, scenario by scenario: a design space evaluated at its starting
        values. None is the boat with no appendages.
    solver : Iteration limit, attitude bounds and threads (``Solver``).

    Returns
    -------
    A dict with ``status`` ("converged", "iteration limit", "infeasible", or the
    solver's own word), ``design`` (every design variable), ``scenarios`` (for each:
    weight, mass, centre of gravity, quantities, utility, violations), ``expected``
    (weighted mean and standard deviation of the utility and of each quantity),
    ``objective``, ``references`` and ``notes``. The quantities are re-evaluated at
    the solution with the hull integrated directly, each scenario at its own
    balance. A solve that does not converge still returns the design it reached.

    The boat model is built once, as one function, and applied to every scenario,
    so memory does not grow with their number; time does.
    """
    solver = solver or Solver()
    t0 = time.time()
    names, low, high, start = _all_variables(boat, design)
    scenario_list, notes = load_scenarios(boat, scenarios)
    n, n_design = len(scenario_list), len(names)
    placed_names = [load.name for load in boat.loads if load.placed]
    tables = hull_tables(boat, velocity, solver.trim_bounds, jobs=solver.table_jobs)
    start_values = dict(zip(names, start))

    ### Each scenario's balance for the starting design: starting states, and references.
    states0, start_quantities = [], []
    for s in scenario_list:
        mass, cg = centre_of_gravity(boat, start_values, s.mass, s.moment, s.placed_masses)
        try:
            sinkage, trim = balance(boat, velocity, design, start_values, mass, cg, propulsion, tables)
        except RuntimeError as error:
            raise RuntimeError(f"The starting design has no balance in scenario {s.name!r}: {error}") from error
        states0.append((sinkage, float(onp.clip(trim, *solver.trim_bounds))))
        q, _, _ = quantities(boat, velocity, design, start_values, mass, cg, sinkage, trim, propulsion, tables)
        start_quantities.append(as_numbers(q))
    for i, c in enumerate(objective.costs):
        if c.exponent != 0 and any(float(_cost_value(c, q)) <= 0 for q in start_quantities):
            raise ValueError(f"Cost {i} is not positive for the starting design; a cost must be a positive quantity.")
    need = objective.needed()
    base = baseline_quantities(boat, velocity, baseline, propulsion, scenarios) if _needs_baseline(objective) else None
    references = _references(objective, start_quantities[0], base, n)
    measure_scale = _measure_scale(objective, start_quantities[0])
    _, _, start_objective = _summarize(objective, scenario_list, start_quantities, references, measure_scale)

    ### The model, once, as a function of (design, state, scenario) that returns only
    ### what the solver needs: the two balance residuals and the log-utility. (Were it
    ### to return every quantity, the solver would differentiate every one of them.)
    n_costs = len(objective.costs)

    def scenario_outputs(d, s_, p):
        values = {name: d[i] for i, name in enumerate(names)}
        mass, cg = centre_of_gravity(
            boat, values, p[0], [p[1], p[2], p[3]], {name: p[4 + j] for j, name in enumerate(placed_names)}
        )
        q, r_force, r_moment = quantities(
            boat, velocity, design, values, mass, cg, s_[0], s_[1], propulsion, tables, need=need
        )
        symbolic_references = {i: p[4 + len(placed_names) + i] for i in range(n_costs)}
        return cas.vertcat(r_force, r_moment, objective.log_utility(q, symbolic_references, measure_scale))

    design_vars = cas.MX.sym("x_design", max(n_design, 1))
    state_vars = cas.MX.sym("x_state", 2, n)
    parameters = cas.DM(onp.array([
        s.parameters(placed_names) + [references[k][i] for i in range(n_costs)]
        for k, s in enumerate(scenario_list)
    ]).T)
    if solver.inline:
        outputs = cas.horzcat(*[scenario_outputs(design_vars, state_vars[:, k], parameters[:, k]) for k in range(n)])
    else:
        d = cas.MX.sym("design", max(n_design, 1))
        s_ = cas.MX.sym("state", 2)
        p = cas.MX.sym("scenario", parameters.shape[0])
        one = cas.Function("scenario", [d, s_, p], [scenario_outputs(d, s_, p)])
        mapped = one.map(n, "thread", solver.threads) if solver.threads > 1 else one.map(n)
        outputs = mapped(design_vars, state_vars, parameters)

    # The objective, in logarithms: ln(sum_k w_k U_k), less the risk term. The same
    # design maximizes it, and it keeps a slope where the utility itself underflows.
    def log_objective(log_utilities):
        logs = [log_utilities[k] + onp.log(scenario_list[k].weight) for k in range(n)]
        log_mean = cas.logsumexp(cas.vertcat(*logs)) if n > 1 else logs[0]
        if objective.risk_aversion == 0 or n == 1:
            return log_mean
        relative = [cas.exp(l - onp.log(s.weight) - log_mean) for l, s in zip(logs, scenario_list)]
        spread = cas.sqrt(sum(s.weight * (u - 1.0) ** 2 for u, s in zip(relative, scenario_list)) + 1e-12)
        factor = 1.0 - objective.risk_aversion * spread
        return log_mean + cas.log(0.5 * (factor + cas.sqrt(factor**2 + 1e-6)))

    x = cas.vertcat(design_vars, cas.vec(state_vars))
    residuals, log_utilities = cas.vec(outputs[:2, :]), outputs[2, :].T
    nlp = {"x": x, "f": -log_objective(log_utilities), "g": residuals}

    # The derivatives are supplied, taken in forward mode. For this model (whose
    # stiffness is itself a derivative) a forward sweep costs a fraction of a second
    # per variable and a reverse sweep tens of seconds; left to itself the solver
    # would take the objective's gradient in reverse.
    everything = cas.vertcat(residuals, log_utilities)
    jacobian = cas.jacobian(everything, x, {"allow_reverse": False})
    symbol = cas.MX.sym("log_utilities", n)
    weights_of_logs = cas.Function("d_objective", [symbol], [cas.gradient(-log_objective(symbol), symbol)])
    no_parameters = cas.MX.sym("p", 0)
    grad_f = cas.Function(
        "nlp_grad_f", [x, no_parameters],
        [nlp["f"], jacobian[2 * n:, :].T @ weights_of_logs(log_utilities)], ["x", "p"], ["f", "grad_f_x"],
    )
    jac_g = cas.Function(
        "nlp_jac_g", [x, no_parameters], [residuals, jacobian[: 2 * n, :]], ["x", "p"], ["g", "jac_g_x"]
    )
    options = {
        "grad_f": grad_f,
        "jac_g": jac_g,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.max_iter": solver.max_iterations,
        "ipopt.print_level": 5 if solver.verbose else 0,
        "print_time": bool(solver.verbose),
    }
    if solver.max_seconds is not None:
        options["ipopt.max_wall_time"] = float(solver.max_seconds)
    if solver.progress:
        n_x, n_g = nlp["x"].shape[0], nlp["g"].shape[0]

        class Progress(cas.Callback):
            def __init__(self):
                cas.Callback.__init__(self)
                self.count, self.started = 0, time.time()
                self.construct("progress", {})

            def get_n_in(self):
                return cas.nlpsol_n_out()

            def get_n_out(self):
                return 1

            def get_sparsity_in(self, i):
                size = {"x": n_x, "lam_x": n_x, "g": n_g, "lam_g": n_g, "f": 1}.get(cas.nlpsol_out(i), 0)
                return cas.Sparsity.dense(size)

            def eval(self, arg):
                named = dict(zip(cas.nlpsol_out(), arg))
                print(
                    f"  iteration {self.count:3d}: objective {float(onp.exp(-float(named['f']))):.6g}, "
                    f"largest residual {float(onp.max(onp.abs(onp.array(named['g'])))):.1e}, "
                    f"{time.time() - self.started:.0f} s",
                    flush=True,
                )
                self.count += 1
                return [0]

        progress = Progress()  # (kept alive for the solve)
        options["iteration_callback"] = progress
        print(f"  model built and starting point balanced in {time.time() - t0:.0f} s; {n} scenario(s), "
              f"{n_design} design variable(s)", flush=True)
    ipopt = cas.nlpsol("design", "ipopt", nlp, options)
    pad = [0.0] if n_design == 0 else []
    x0 = start + pad + [v for state in states0 for v in state]
    lbx = low + pad + [solver.sinkage_bounds[0], solver.trim_bounds[0]] * n
    ubx = high + pad + [solver.sinkage_bounds[1], solver.trim_bounds[1]] * n
    solution = ipopt(x0=x0, lbx=lbx, ubx=ubx, lbg=0.0, ubg=0.0)
    stats = ipopt.stats()
    x = onp.array(solution["x"]).ravel()
    solved = {name: float(x[i]) for i, name in enumerate(names)}
    offset = max(n_design, 1)
    guesses = [(float(x[offset + 2 * k]), float(x[offset + 2 * k + 1])) for k in range(n)]

    ### Assess the solution with the hull integrated directly.
    result = evaluate(
        boat, velocity, design, solved, objective, propulsion, scenarios,
        references=references, measure_scale=measure_scale, guesses=guesses,
    )
    if base is not None:
        result["baseline"] = base
    result.update(
        status=_STATUS.get(stats["return_status"], stats["return_status"]),
        iterations=int(stats["iter_count"]),
        seconds=time.time() - t0,
        start={"design": start_values, "objective": start_objective},
        solver_objective=float(onp.exp(-float(solution["f"]))),
        residual=float(onp.max(onp.abs(onp.array(solution["g"])))),
        solver_seconds={k[7:]: float(v) for k, v in stats.items() if k.startswith("t_wall_")},
    )
    return result
