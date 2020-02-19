import numpy as np
import pytest
from orix.base import DimensionError
from orix.quaternion import Quaternion, check_quaternion
from orix.vector import Vector3d

values = [
    (0.707, 0.0, 0.0, 0.707),
    (0.5, -0.5, -0.5, 0.5),
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
    ((0.5, -0.5, -0.5, 0.5), (0.0, 0.0, 0.0, 1.0),),
    Quaternion(
        [
            [(0.0, 0.0, 0.0, 1.0), (0.707, 0.0, 0.0, 0.707),],
            [(1.0, 1.0, 1.0, 1.0), (0.707, 0.0, 0.0, 0.707),],
        ]
    ),
    np.array((4, 3, 2, 1)),
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
    ((1, 0, 0.5, 0), (3, 1, -1, -2),),
    [
        [[0.343, 0.343, 0, -0.333], [-7, -8, -9, -10],],
        [[0.00001, -0.0001, 0.001, -0.01], [0, 0, 0, 0]],
    ],
]


@pytest.fixture(params=singles)
def something(request):
    return Quaternion(request.param)


@pytest.mark.parametrize("input_length", [1, 2, 3, 5, 6, 8,])
def test_init(input_length):
    with pytest.raises(DimensionError):
        Quaternion(tuple(range(input_length)))


def test_neg(quaternion):
    assert np.all((-quaternion).data == -(quaternion.data))


def test_norm(quaternion):
    assert np.all(quaternion.norm.data == (quaternion.data ** 2).sum(axis=-1) ** 0.5)


def test_unit(quaternion):
    assert np.allclose(quaternion.unit.norm.data, 1)


def test_conj(quaternion):
    assert np.all(quaternion.data[..., 0] == quaternion.conj.data[..., 0])
    assert np.all(quaternion.data[..., 1:] == -quaternion.conj.data[..., 1:])


def test_mul(quaternion, something):
    sa, sb, sc, sd = (
        something.a.data,
        something.b.data,
        something.c.data,
        something.d.data,
    )
    qa, qb, qc, qd = (
        quaternion.a.data,
        quaternion.b.data,
        quaternion.c.data,
        quaternion.d.data,
    )
    q1 = quaternion * something
    assert isinstance(q1, Quaternion)
    assert np.allclose(q1.a.data, sa * qa - sb * qb - sc * qc - sd * qd)
    assert np.allclose(q1.b.data, qa * sb + qb * sa + qc * sd - qd * sc)
    assert np.allclose(q1.c.data, qa * sc - qb * sd + qc * sa + qd * sb)
    assert np.allclose(q1.d.data, qa * sd + qb * sc - qc * sb + qd * sa)


def test_mul_identity(quaternion, identity):
    assert np.all((quaternion * identity).data == quaternion.data)


def test_no_multiplicative_inverse(quaternion, something):
    q1 = quaternion * something
    q2 = something * quaternion
    assert np.all(q1 != q2)


def test_inverse(quaternion):
    assert np.allclose((quaternion * ~quaternion).data, (~quaternion * quaternion).data)
    assert np.allclose((quaternion * ~quaternion).a.data, 1)
    assert np.allclose((quaternion * ~quaternion).data[..., 1:], 0)


def test_dot(quaternion, something):
    assert np.all(
        quaternion.dot(quaternion).data == np.sum(quaternion.data ** 2, axis=-1)
    )
    assert np.all(quaternion.dot(something).data == something.dot(quaternion).data)


def test_dot_outer(quaternion, something):
    d = quaternion.dot_outer(something)
    assert d.shape == quaternion.shape + something.shape
    for i in np.ndindex(quaternion.shape):
        for j in np.ndindex(something.shape):
            assert np.allclose(d[i + j].data, quaternion[i].dot(something[j]).data)


@pytest.mark.parametrize(
    "quaternion, vector, expected",
    [
        ((0.5, 0.5, 0.5, 0.5), (1, 0, 0), (0, 1, 0)),
        ((np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2), (0, 1, 0), (-1, 0, 0)),
        ((0, 1, 0, 0), (0, 1, 0), (0, -1, 0)),
        (
            (0, np.sqrt(3) / 3, np.sqrt(3) / 3, -np.sqrt(3) / 3),
            (1, 1, 0),
            (1 / 3, 1 / 3, -4 / 3),
        ),
    ],
)
def test_multiply_vector(quaternion, vector, expected):
    q = Quaternion(quaternion)
    v = Vector3d(vector)
    v_new = q * v
    assert np.allclose(v_new.data, expected)


def test_check_quat():
    """ check is an oddly named function"""
    quat = Quaternion([2, 2, 2, 2])
    assert np.allclose(quat.data, check_quaternion(quat).data)


def test_abcd():
    quat = Quaternion([2, 2, 2, 2])
    quat.a = 1
    quat.b = 1
    quat.c = 1
    quat.d = 1
    assert np.allclose(quat.data, 1)


def test_mean(quaternion):
    _ = quaternion.mean()
    return None


def test_antipodal(quaternion):
    _ = quaternion.antipodal
    return None


@pytest.mark.xfail(strict=True, reason="NotImplemented")
def test_edgecase_outer(quaternion):
    threetwoq = quaternion.outer([3, 2])


@pytest.mark.xfail(strict=True, reason="NotImplemented")
def test_failing_mul(quaternion):
    quaternion * "cant-mult-by-this"
