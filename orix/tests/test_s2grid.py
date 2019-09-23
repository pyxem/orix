import pytest

from orix.grid.s1grid import S1Grid
from orix.grid.s2grid import S2Grid


@pytest.fixture
def s1grid(request):
    return S1Grid(request.param)


@pytest.fixture
def s2grid(request):
    tg, rg = request.param
    return S2Grid(tg, rg)
