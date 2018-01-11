import numpy as np
import pytest

from texpy.quaternion.quaternion import Quaternion
from texpy.vector.vector3d import Vector3d

values = [
    (0.707, 0., 0., 0.707),
    (0.5, -0.5, -0.5, 0.5),
    (0., 0., 0., 1.),
    (1., 1., 1., 1.),
    (
        (0.5, -0.5, -0.5, 0.5),
        (0., 0., 0., 1.),
    ),
    Quaternion([
        [(0., 0., 0., 1.), (0.707, 0., 0., 0.707),],
        [(1., 1., 1., 1.), (0.707, 0., 0., 0.707),]
    ]),
    np.array((4, 3, 2, 1))
]


@pytest.fixture(params=values)
def quaternion(request):
    return Quaternion(request.param)


@pytest.fixture
def identity():
    return Quaternion((1, 0, 0, 0))

singles = [
    (0.881, 0.665, 0.123, 0.517),
    (0.111, 0.222, 0.333, 0.444),
    (
        (1, 0, 0.5, 0),
        (3, 1, -1, -2),
    ),
    [
        [[0.343, 0.343, 0, -0.333], [-7, -8, -9, -10],],
        [[0.00001, -0.0001, 0.001, -0.01], [0, 0, 0, 0]]
    ]
]


@pytest.fixture(params=singles)
def something(request):
    return Quaternion(request.param)


@pytest.mark.parametrize("input_length", [
    1, 2, 3, 5, 6, 8,
])
def test_init(input_length):
    with pytest.raises(AssertionError):
        Quaternion(tuple(range(input_length)))


def test_neg(quaternion):
    assert np.all(-quaternion.data == -(quaternion.data))


def test_norm(quaternion):
    assert np.all(quaternion.norm == (quaternion.data**2).sum(axis=-1)**0.5)


def test_unit(quaternion):
    assert np.allclose(quaternion.unit.norm, 1)


def test_conj(quaternion):
    assert np.all(quaternion.data[..., 0] == quaternion.conj.data[..., 0])
    assert np.all(quaternion.data[..., 1:] == -quaternion.conj.data[..., 1:])


def test_mul(quaternion, something):
    sa, sb, sc, sd = something.a, something.b, something.c, something.d
    qa, qb, qc, qd = quaternion.a, quaternion.b, quaternion.c, quaternion.d
    q1 = quaternion * something
    assert isinstance(q1, Quaternion)
    assert np.allclose(q1.a, sa * qa - sb * qb - sc * qc - sd * qd)
    assert np.allclose(q1.b, qa * sb + qb * sa + qc * sd - qd * sc)
    assert np.allclose(q1.c, qa * sc - qb * sd + qc * sa + qd * sb)
    assert np.allclose(q1.d, qa * sd + qb * sc - qc * sb + qd * sa)


def test_mul_identity(quaternion, identity):
    assert np.all((quaternion * identity).data == quaternion.data)


def test_no_multiplicative_inverse(quaternion, something):
    q1 = quaternion * something
    q2 = something * quaternion
    assert np.all(q1 != q2)


def test_inverse(quaternion):
    assert np.allclose((quaternion * ~quaternion).data, (~quaternion * quaternion).data)
    assert np.allclose((quaternion * ~quaternion).a, 1)
    assert np.allclose((quaternion * ~quaternion).data[..., 1:], 0)


def test_dot(quaternion, something):
    assert np.all(quaternion.dot(quaternion).data == np.sum(quaternion.data ** 2, axis=-1))
    assert np.all(quaternion.dot(something).data == something.dot(quaternion).data)


def test_dot_outer(quaternion, something):
    d = quaternion.dot_outer(something)
    assert d.shape == quaternion.shape + something.shape
    for i in np.ndindex(quaternion.shape):
        for j in np.ndindex(something.shape):
            assert np.allclose(d[i + j], quaternion[i].dot(something[j]))


def test_angle_with_self(quaternion):
    a = quaternion.angle_with(quaternion)
    assert np.allclose(a, 0, atol=1e-6)


def test_angle_with(quaternion, something):
    a = quaternion.angle_with(something)
    assert np.all(a >= 0)
    assert np.all(a <= np.pi)


def test_axis(quaternion):
    axis = quaternion.axis
    assert isinstance(axis, Vector3d)
    assert np.all(axis.norm == 1)


def test_angle(quaternion):
    angle = quaternion.angle
    assert np.all(angle >= 0)
    assert np.all(angle <= np.pi)


@pytest.mark.parametrize('quaternion, vector, expected', [
    ((0.5, 0.5, 0.5, 0.5), (1, 0, 0), (0, 1, 0)),
    ((np.sqrt(2)/2, 0, 0, np.sqrt(2)/2), (0, 1, 0), (-1, 0, 0)),
    ((0, 1, 0, 0), (0, 1, 0), (0, -1, 0)),
    ((0, np.sqrt(3)/3, np.sqrt(3)/3, -np.sqrt(3)/3), (1, 1, 0), (1/3, 1/3, -4/3)),

])
def test_multiply_vector(quaternion, vector, expected):
    q = Quaternion(quaternion)
    v = Vector3d(vector)
    v_new = q * v
    assert np.allclose(v_new.data, expected)