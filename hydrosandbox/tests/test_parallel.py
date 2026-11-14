import math
import os
import pytest
from hydrosandbox.parallel import default_jobs, parallel_map


def test_parallel_map_keeps_order_and_unpacks_tasks():
    assert default_jobs() >= 1
    numbers = list(range(1, 9))
    # Single arguments, in task order, across processes and in-process.
    assert parallel_map(math.sqrt, numbers, jobs=2) == [math.sqrt(n) for n in numbers]
    assert parallel_map(math.sqrt, numbers, jobs=1) == [math.sqrt(n) for n in numbers]
    # Tuples are positional arguments; dicts are keyword arguments.
    assert parallel_map(pow, [(2, 3), (3, 2)], jobs=2) == [8, 9]
    assert parallel_map(complex, [dict(real=1.0, imag=2.0), dict(real=3.0)], jobs=2) == [
        1 + 2j,
        3 + 0j,
    ]
    assert parallel_map(math.sqrt, [], jobs=4) == []


def test_parallel_map_reraises_and_restores_the_thread_settings():
    before = os.environ.get("OMP_NUM_THREADS")
    with pytest.raises(TypeError):
        parallel_map(pow, [(2, 3), ("a", 2)], jobs=2)
    assert os.environ.get("OMP_NUM_THREADS") == before


def test_parallel_map_reports_progress(capsys):
    seen = []
    results = parallel_map(
        math.sqrt, [4.0, 9.0, 16.0], jobs=2,
        progress=lambda n_done, n_total, index, result, seconds: seen.append(
            (n_done, n_total, index, result)
        ),
    )
    assert results == [2.0, 3.0, 4.0]
    assert [s[0] for s in seen] == [1, 2, 3] and all(s[1] == 3 for s in seen)
    assert sorted((s[2], s[3]) for s in seen) == [(0, 2.0), (1, 3.0), (2, 4.0)]

    parallel_map(math.sqrt, [4.0, 9.0], jobs=1, progress=True, labels=["first", "second"])
    err = capsys.readouterr().err
    assert "[1/2] done: first" in err and "[2/2] done: second" in err


def test_parallel_map_jobs_one_pins_thread_vars_and_restores_env():
    before = os.environ.get("OPENBLAS_NUM_THREADS")
    seen = parallel_map(
        str,
        [1],
        jobs=1,
    )
    assert seen == ["1"]
    seen_vars = parallel_map(
        os.environ.get,
        ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"],
        jobs=1,
    )
    assert seen_vars == ["1", "1"]
    assert os.environ.get("OPENBLAS_NUM_THREADS") == before
