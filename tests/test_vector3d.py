from math import pi
import numpy as np
import pytest

from texpy.vector.vector3d import Vector3d
from texpy.scalar.scalar import Scalar

vectors = [
    (1, 0, 0),
    (0, 0, 1),
    (
        (0.5, 0.5, 0.5),
        (-1, 0, 0),
    ),
    [
        [[-0.707, 0.707, 1], [2, 2, 2]],
        [[0.1, -0.3, 0.2], [-5, -6, -7]],
    ],
    np.random.rand(3)
]

singles = [
    (1, -1, 1),
    (-5, -5, -6),
    [
        [9, 9, 9],
        [0.001, 0.0001, 0.00001],
    ],
    np.array([
        [[0.5, 0.25, 0.125], [-0.125, 0.25, 0.5]],
        [[1, 2, 4], [1, -0.3333, 0.1667]],
    ])
]

numbers = [-12, 0.5, -0.333333333, 4]


@pytest.fixture(params=vectors)
def vector(request):
    return Vector3d(request.param)

@pytest.fixture(params=numbers)
def scalar(request):
    return Scalar(request.param)


@pytest.fixture(params=singles)
def something(request):
    return Vector3d(request.param)


@pytest.fixture(params=numbers)
def number(request):
    return request.param


def test_neg(vector):
    assert np.all((-vector).data == -(vector.data))


def test_add_number(vector, number):
    assert np.all((vector + number).data == vector.data + number)
    assert np.all((number + vector).data == (vector + number).data)


def test_add_vector(vector, something):
    assert np.all((vector + something).data == vector.data + something.data)
    assert np.all((vector + something).data == (something + vector).data)


@pytest.mark.parametrize('vector, scalar, expected', [
    ((1, 0, 0), 1, (2, 1, 1)),
    (((-1, -1, 1), (0.5, 0.5, -0.5)), (-1, 2), ((-2, -2, 0), (2.5, 2.5, 1.5))),
], indirect=['vector', 'scalar'])
def test_add_scalar(vector, scalar, expected):
    sum = vector + scalar
    assert np.allclose(sum.data, expected)


def test_sub_number(vector, number):
    assert np.all((vector - number).data == vector.data - number)
    assert np.all((number - vector).data == -(vector - number).data)


def test_sub_vector(vector, something):
    assert np.all((vector - something).data == vector.data - something.data)
    assert np.all((vector - something).data == -(something - vector).data)


@pytest.mark.parametrize('vector, scalar, expected', [
    ((1, 0, 0), 1, (0, -1, -1)),
    (((-1, -1, 1), (0.5, 0.5, -0.5)), (-1, 2), ((0, 0, 2), (-1.5, -1.5, -2.5))),
], indirect=['vector', 'scalar'])
def test_sub_scalar(vector, scalar, expected):
    sub = vector - scalar
    assert np.allclose(sub.data, expected)


def test_mul_number(vector, number):
    assert np.all((vector * number).data == vector.data * number)
    assert np.all((number * vector).data == (vector * number).data)


def test_mul_error(vector, something):
    with pytest.raises(ValueError):
        vector * something


@pytest.mark.parametrize('vector, scalar, expected', [
    ((1, 0, 0), 1, (1, 0, 0)),
    (((-1, -1, 1), (0.5, 0.5, -0.5)), (-1, 2), ((1, 1, -1), (1, 1, -1))),
], indirect=['vector', 'scalar'])
def test_mul_scalar(vector, scalar, expected):
    mul = vector * scalar
    assert np.allclose(mul.data, expected)


def test_dot(vector, something):
    assert np.allclose(vector.dot(vector).data, (vector.data ** 2).sum(axis=-1))
    assert np.allclose(vector.dot(something).data, something.dot(vector).data)


def test_dot_error(vector, number):
    with pytest.raises(ValueError):
        vector.dot(number)


def test_dot_outer(vector, something):
    d = vector.dot_outer(something).data
    assert d.shape == vector.shape + something.shape
    for i in np.ndindex(vector.shape):
        for j in np.ndindex(something.shape):
            assert np.allclose(d[i + j], vector[i].dot(something[j]).data)


def test_cross(vector, something):
    assert isinstance(vector.cross(something), Vector3d)


def test_cross_error(vector, number):
    with pytest.raises(AttributeError):
        vector.cross(number)


@pytest.mark.parametrize('theta, phi, r, expected', [
    (np.pi/4, np.pi/4, 1, Vector3d((0.5, 0.5, 0.707107))),
    (2 * np.pi / 3, 7 * np.pi / 6, 1, Vector3d((-0.75, -0.433013, -0.5))),
])
def test_polar(theta, phi, r, expected):
    assert np.allclose(Vector3d.from_polar(theta, phi, r).data, expected.data, atol=1e-5)

@pytest.mark.parametrize('shape', [
    (1,),
    (2, 2),
    (5, 4, 3),
])
def test_zero(shape):
    v = Vector3d.zero(shape)
    assert v.shape == shape
    assert v.data.shape[-1] == v.dim

def test_angle_with(vector, something):
    a = vector.angle_with(vector).data
    assert np.allclose(a, 0)
    a = vector.angle_with(something).data
    assert np.all(a >= 0)
    assert np.all(a <= np.pi)


def test_mul_array(vector):
    array = np.random.rand(*vector.shape)
    m1 = vector * array
    m2 = array * vector
    assert isinstance(m1, Vector3d)
    assert isinstance(m2, Vector3d)
    assert np.all(m1.data == m2.data)


@pytest.mark.parametrize('vector, rotation, expected', [
    ((1, 0, 0), pi / 2, (0, 1, 0)),
    ((1, 1, 0), pi / 2, (-1, 1, 0)),
    ((1, 1, 1), -pi / 2, (1, -1, 1)),
])
def test_rotate(vector, rotation, expected):
    r = Vector3d(vector).rotate(Vector3d.zvector(), rotation)
    assert isinstance(r, Vector3d)
    assert np.allclose(r.data, expected)