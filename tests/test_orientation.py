import pytest
import numpy as np

from texpy.quaternion.orientation import Orientation
from texpy.quaternion.symmetry import *


@pytest.mark.parametrize('shape, symmetry', [
    ((6,), C4),
    ((3, 3), C4),
    ((3, 3), C3v),
])
def test_set_symmetry_works(shape, symmetry):
    o = Orientation.random(shape)
    oe = o.set_symmetry(symmetry)
    assert oe.shape == o.shape


@pytest.mark.parametrize('symmetry', [
    C1, C2, C4, C4v, C3, S6, D3, D6h, T, O, Oh
])
def test_set_symmetry(symmetry):
    o = Orientation.random(10)
    os = o.set_symmetry(symmetry)
    for i, o_i in enumerate(o):
        os_i = o_i.set_symmetry(symmetry)
        assert np.allclose(os_i.data, os[i].data)