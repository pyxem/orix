import pytest
import numpy as np

from orix.vector import Vector3d
from orix.vector.spherical_region import SphericalRegion


@pytest.fixture(params=[(0, 0, 1)])
def spherical_region(request):
    return SphericalRegion(request.param)


@pytest.fixture(params=[(0, 0, 1)])
def vector(request):
    return Vector3d(request.param)


@pytest.mark.parametrize(
    "spherical_region, vector, expected",
    [
        ([0, 0, 1], [[0, 0, 0.5], [0, 0, -0.5], [0, 1, 0]], [True, False, False]),
        ([[0, 0, 1], [0, 1, 0]], [[0, 1, 1], [0, 0, 1]], [True, False]),
    ],
    indirect=["spherical_region", "vector"],
)
def test_gt(spherical_region, vector, expected):
    inside = vector < spherical_region
    assert np.all(np.equal(inside, expected))


@pytest.mark.parametrize(
    "spherical_region, vector, expected",
    [
        ([0, 0, 1], [[0, 0, 0.5], [0, 0, -0.5], [0, 1, 0]], [True, False, True]),
        ([[0, 0, 1], [0, 1, 0]], [[0, 1, 1], [0, 0, 1]], [True, True]),
    ],
    indirect=["spherical_region", "vector"],
)
def test_ge(spherical_region, vector, expected):
    inside = vector <= spherical_region
    assert np.all(np.equal(inside, expected))
