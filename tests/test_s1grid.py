import pytest
import numpy as np
from math import cos, sin, pi
from orix.grid.s1grid import S1Grid


@pytest.fixture
def s1grid(request):
    return S1Grid(request.param)


@pytest.mark.parametrize('s1grid, minimum, maximum', [
    (np.radians(np.linspace(0, 45, 10)), 0, pi/4),
    (np.radians(np.linspace(-45, 45, 10)), -pi/4, pi/4),
], indirect=['s1grid'])
def test_min_max(s1grid, minimum, maximum):
    assert s1grid.minimum == minimum
    assert s1grid.maximum == maximum