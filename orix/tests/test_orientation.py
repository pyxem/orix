import numpy as np
import pytest

from orix.quaternion.orientation import Orientation, Misorientation
from orix.quaternion.symmetry import C1, C2, C3, C4, D2, D3, D6, T, O
from orix.vector import Vector3d


@pytest.fixture
def vector(request):
    return Vector3d(request.param)


@pytest.fixture(params=[(0.5, 0.5, 0.5, 0.5), (0.5 ** 0.5, 0, 0, 0.5 ** 0.5)])
def orientation(request):
    return Orientation(request.param)


@pytest.mark.parametrize(
    "orientation, symmetry, expected",
    [
        ([(1, 0, 0, 0)], C1, [(1, 0, 0, 0)]),
        ([(1, 0, 0, 0)], C4, [(1, 0, 0, 0)]),
        ([(1, 0, 0, 0)], D3, [(1, 0, 0, 0)]),
        ([(1, 0, 0, 0)], T, [(1, 0, 0, 0)]),
        ([(1, 0, 0, 0)], O, [(1, 0, 0, 0)]),
        # 7pi/12 -C2-> # 7pi/12
        ([(0.6088, 0, 0, 0.7934)], C2, [(-0.7934, 0, 0, 0.6088)]),
        # 7pi/12 -C3-> # 7pi/12
        ([(0.6088, 0, 0, 0.7934)], C3, [(-0.9914, 0, 0, 0.1305)]),
        # 7pi/12 -C4-> # pi/12
        ([(0.6088, 0, 0, 0.7934)], C4, [(-0.9914, 0, 0, -0.1305)]),
        # 7pi/12 -O-> # pi/12
        ([(0.6088, 0, 0, 0.7934)], O, [(-0.9914, 0, 0, -0.1305)]),
    ],
    indirect=["orientation"],
)
def test_set_symmetry(orientation, symmetry, expected):
    o = orientation.set_symmetry(symmetry)
    assert np.allclose(o.data, expected, atol=1e-3)


@pytest.mark.parametrize(
    "symmetry, vector",
    [(C1, (1, 2, 3)), (C2, (1, -1, 3)), (C3, (1, 1, 1)), (O, (0, 1, 0))],
    indirect=["vector"],
)
def test_orientation_persistence(symmetry, vector):
    v = symmetry.outer(vector).flatten()
    o = Orientation.random()
    oc = o.set_symmetry(symmetry)
    v1 = o * v
    v1 = Vector3d(v1.data.round(4))
    v2 = oc * v
    v2 = Vector3d(v2.data.round(4))
    assert v1._tuples == v2._tuples


@pytest.mark.parametrize(
    "orientation, symmetry, expected",
    [
        ((1, 0, 0, 0), C1, [0]),
        ([(1, 0, 0, 0), (0.7071, 0.7071, 0, 0)], C1, [[0, np.pi / 2], [np.pi / 2, 0]]),
        ([(1, 0, 0, 0), (0.7071, 0.7071, 0, 0)], C4, [[0, np.pi / 2], [np.pi / 2, 0]]),
        ([(1, 0, 0, 0), (0.7071, 0, 0, 0.7071)], C4, [[0, 0], [0, 0]]),
        (
            [
                [(1, 0, 0, 0), (0.7071, 0, 0, 0.7071)],
                [(0, 0, 0, 1), (0.9239, 0, 0, 0.3827)],
            ],
            C4,
            [
                [[[0, 0], [0, np.pi / 4]], [[0, 0], [0, np.pi / 4]]],
                [[[0, 0], [0, np.pi / 4]], [[np.pi / 4, np.pi / 4], [np.pi / 4, 0]]],
            ],
        ),
    ],
    indirect=["orientation"],
)
def test_distance(orientation, symmetry, expected):
    o = orientation.set_symmetry(symmetry)
    distance = o.distance(verbose=True)
    assert np.allclose(distance, expected, atol=1e-3)


@pytest.mark.parametrize("symmetry", [C1, C2, C4, D2, D6, T, O])
def test_getitem(orientation, symmetry):
    o = orientation.set_symmetry(symmetry)
    assert o[0].symmetry._tuples == symmetry._tuples


@pytest.mark.parametrize("Gl", [C4, C2])
def test_equivalent(Gl):
    """ Tests that the property Misorientation.equivalent runs without error,
    use grain_exchange=True as this falls back to grain_exchange=False when
    Gl!=Gr:

    Gl == C4 is grain exchange
    Gl == C2 is no grain exchange
    """
    m = Misorientation([1, 1, 1, 1])  # any will do
    m_new = m.set_symmetry(Gl, C4, verbose=True)
    m_new.symmetry
    _m = m_new.equivalent(grain_exchange=True)


def test_repr():
    m = Misorientation([1, 1, 1, 1])  # any will do
    print(m)  # hits __repr__
    return None


def test_sub():
    m = Orientation([1, 1, 1, 1])  # any will do
    m.set_symmetry(C4)  # only one as it a O
    _ = m - m  # this should give a set of zeroes
    return None


@pytest.mark.xfail(strict=True, reason=TypeError)
def test_sub_orientation_and_other():
    m = Orientation([1, 1, 1, 1])  # any will do
    _ = m - 3
