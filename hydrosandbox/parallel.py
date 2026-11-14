"""
Running independent analyses on several cores.

Each ``asb.Opti`` solve and each equilibrium solve is single-threaded, so the
natural unit of parallelism is the whole analysis: the starting points of a
multi-start optimization, the points of a parameter sweep, the floors of a
trade-off study. ``parallel_map`` runs such tasks in separate processes.
"""

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence

_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def default_jobs() -> int:
    """The number of worker processes used when ``jobs`` is not given: all cores."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on every platform
        return max(1, os.cpu_count() or 1)


def _call(packed: tuple[Callable[..., Any], Any]) -> Any:
    function, task = packed
    if isinstance(task, dict):
        return function(**task)
    if isinstance(task, tuple):
        return function(*task)
    return function(task)


def parallel_map(
    function: Callable[..., Any],
    tasks: Iterable[Any],
    jobs: int | None = None,
    progress: bool | Callable[[int, int, int, Any, float], None] = False,
    labels: Sequence[str] | None = None,
) -> list[Any]:
    """
    Apply ``function`` to every task, in separate processes, and return the results
    in task order.

    A task that is a ``dict`` is passed as keyword arguments, a ``tuple`` as
    positional arguments, anything else as the single argument.

    ``function``, the tasks and the results must be picklable: ``function`` has to
    be importable (defined at the top level of a module, not a lambda or a nested
    function), and a script that calls this must guard its entry point with
    ``if __name__ == "__main__":``. Hulls, assemblies and plain result dictionaries
    are fine; CasADi expressions are not, so return numbers.

    Workers are started fresh (``spawn``) with numerical libraries held to one
    thread each, so ``jobs`` processes use ``jobs`` cores rather than fighting over
    all of them. With ``jobs=1`` (or a single task) everything runs in this process.

    An exception in any task is re-raised here.

    Progress: with ``progress=True``, one line is written to standard error as each
    task finishes (``[3/20] done: <label> (41 s elapsed)``; ``labels`` names the
    tasks, else their index is used), so a long study shows it is alive and how far
    along it is. Tasks finish in any order; results are still returned in task
    order. Pass a callable instead to receive
    ``(n_done, n_total, task_index, result, seconds_elapsed)`` yourself, e.g. to
    print a result as soon as it exists.
    """
    tasks = list(tasks)
    n_total = len(tasks)
    if jobs is None:
        jobs = default_jobs()
    jobs = max(1, min(int(jobs), n_total))
    started = time.time()

    def report(n_done: int, index: int, result: Any) -> None:
        elapsed = time.time() - started
        if callable(progress):
            progress(n_done, n_total, index, result, elapsed)
        elif progress:
            label = labels[index] if labels is not None else f"task {index}"
            print(
                f"[{n_done}/{n_total}] done: {label} ({elapsed:.0f} s elapsed)",
                file=sys.stderr,
                flush=True,
            )

    saved = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    os.environ.update({name: "1" for name in _THREAD_VARIABLES})
    try:
        if jobs == 1:
            results = []
            for index, task in enumerate(tasks):
                results.append(_call((function, task)))
                report(index + 1, index, results[-1])
            return results

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            futures = {
                pool.submit(_call, (function, task)): index
                for index, task in enumerate(tasks)
            }
            results = [None] * n_total
            for n_done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                results[index] = future.result()
                report(n_done, index, results[index])
            return results
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
