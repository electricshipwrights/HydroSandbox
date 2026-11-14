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
