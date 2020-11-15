from math import pi
import numpy as np
import pytest

from orix.vector import Vector3d, check_vector
from orix.scalar import Scalar

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
    np.random.rand(3),
]

singles = [
    (1, -1, 1),
    (-5, -5, -6),
    [
        [9, 9, 9],
        [0.001, 0.0001, 0.00001],
    ],
    np.array(
        [
            [[0.5, 0.25, 0.125], [-0.125, 0.25, 0.5]],
            [[1, 2, 4], [1, -0.3333, 0.1667]],
        ]
    ),
]

numbers = [-12, 0.5, -0.333333333, 4]


@pytest.fixture(params=vectors)
def vector(request):
    return Vector3d(request.param)


@pytest.fixture(params=singles)
def something(request):
    return Vector3d(request.param)


@pytest.fixture(params=numbers)
def number(request):
    return request.param


def test_check_vector():
    vector3 = Vector3d([2, 2, 2])
    assert np.allclose(vector3.data, check_vector(vector3).data)


def test_neg(vector):
    assert np.all((-vector).data == -(vector.data))


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        ([1, 2, 3], Vector3d([[1, 2, 3], [-3, -2, -1]]), [[2, 4, 6], [-2, 0, 2]]),
        ([1, 2, 3], Scalar([4]), [5, 6, 7]),
        ([1, 2, 3], 0.5, [1.5, 2.5, 3.5]),
        ([1, 2, 3], [-1, 2], [[0, 1, 2], [3, 4, 5]]),
        ([1, 2, 3], np.array([-1, 1]), [[0, 1, 2], [2, 3, 4]]),
        pytest.param([1, 2, 3], "dracula", None, marks=pytest.mark.xfail),
    ],
    indirect=["vector"],
)
def test_add(vector, other, expected):
    s1 = vector + other
    s2 = other + vector
    assert np.allclose(s1.data, expected)
    assert np.allclose(s1.data, s2.data)


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        ([1, 2, 3], Vector3d([[1, 2, 3], [-3, -2, -1]]), [[0, 0, 0], [4, 4, 4]]),
        ([1, 2, 3], Scalar([4]), [-3, -2, -1]),
        ([1, 2, 3], 0.5, [0.5, 1.5, 2.5]),
        ([1, 2, 3], [-1, 2], [[2, 3, 4], [-1, 0, 1]]),
        ([1, 2, 3], np.array([-1, 1]), [[2, 3, 4], [0, 1, 2]]),
        pytest.param([1, 2, 3], "dracula", None, marks=pytest.mark.xfail),
    ],
    indirect=["vector"],
)
def test_sub(vector, other, expected):
    s1 = vector - other
    s2 = other - vector
    assert np.allclose(s1.data, expected)
    assert np.allclose(-s1.data, s2.data)


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        pytest.param(
            [1, 2, 3],
            Vector3d([[1, 2, 3], [-3, -2, -1]]),
            [[0, 0, 0], [4, 4, 4]],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        ([1, 2, 3], Scalar([4]), [4, 8, 12]),
        ([1, 2, 3], 0.5, [0.5, 1.0, 1.5]),
        ([1, 2, 3], [-1, 2], [[-1, -2, -3], [2, 4, 6]]),
        ([1, 2, 3], np.array([-1, 1]), [[-1, -2, -3], [1, 2, 3]]),
        pytest.param([1, 2, 3], "dracula", None, marks=pytest.mark.xfail),
    ],
    indirect=["vector"],
)
def test_mul(vector, other, expected):
    s1 = vector * other
    s2 = other * vector
    assert np.allclose(s1.data, expected)
    assert np.allclose(s1.data, s2.data)


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        pytest.param(
            [1, 2, 3],
            Vector3d([[1, 2, 3], [-3, -2, -1]]),
            [[0, 0, 0], [4, 4, 4]],
            marks=pytest.mark.xfail(raises=ValueError),
        ),
        ([4, 8, 12], Scalar([4]), [1, 2, 3]),
        ([0.5, 1.0, 1.5], 0.5, [1, 2, 3]),
        (
            [1, 2, 3],
            [-1, 2],
            [[-1, -2, -3], [1 / 2, 1, 3 / 2]],
        ),
        (
            [1, 2, 3],
            np.array([-1, 1]),
            [[-1, -2, -3], [1, 2, 3]],
        ),
        pytest.param([1, 2, 3], "dracula", None, marks=pytest.mark.xfail),
    ],
    indirect=["vector"],
)
def test_div(vector, other, expected):
    s1 = vector / other
    assert np.allclose(s1.data, expected)


@pytest.mark.xfail
def test_rdiv():
    v = Vector3d([1, 2, 3])
    other = 1
    _ = other / v


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


@pytest.mark.parametrize(
    "theta, phi, r, expected",
    [
        (np.pi / 4, np.pi / 4, 1, Vector3d((0.5, 0.5, 0.707107))),
        (2 * np.pi / 3, 7 * np.pi / 6, 1, Vector3d((-0.75, -0.433013, -0.5))),
    ],
)
def test_polar(theta, phi, r, expected):
    assert np.allclose(
        Vector3d.from_polar(theta, phi, r).data, expected.data, atol=1e-5
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (2, 2),
        (5, 4, 3),
    ],
)
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


@pytest.mark.parametrize(
    "vector, x, y, z",
    [
        ([1, 2, 3], 1, 2, 3),
        ([[0, 2, 3], [2, 2, 3]], [0, 2], [2, 2], [3, 3]),
    ],
    indirect=["vector"],
)
def test_xyz(vector, x, y, z):
    vx, vy, vz = vector.xyz
    assert np.allclose(vx, x)
    assert np.allclose(vy, y)
    assert np.allclose(vz, z)


@pytest.mark.parametrize(
    "vector, rotation, expected",
    [
        ((1, 0, 0), pi / 2, (0, 1, 0)),
        ((1, 1, 0), pi / 2, (-1, 1, 0)),
        (
            (1, 1, 0),
            [pi / 2, pi, 3 * pi / 2, 2 * pi],
            [(-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0)],
        ),
        ((1, 1, 1), -pi / 2, (1, -1, 1)),
    ],
    indirect=["vector"],
)
def test_rotate(vector, rotation, expected):
    r = Vector3d(vector).rotate(Vector3d.zvector(), rotation)
    assert isinstance(r, Vector3d)
    assert np.allclose(r.data, expected)


@pytest.mark.parametrize(
    "vector, data, expected",
    [
        ([1, 2, 3], 3, [3, 2, 3]),
        ([[0, 2, 3], [2, 2, 3]], 1, [[1, 2, 3], [1, 2, 3]]),
        ([[0, 2, 3], [2, 2, 3]], [-1, 1], [[-1, 2, 3], [1, 2, 3]]),
    ],
    indirect=["vector"],
)
def test_assign_x(vector, data, expected):
    vector.x = data
    assert np.allclose(vector.data, expected)


@pytest.mark.parametrize(
    "vector, data, expected",
    [
        ([1, 2, 3], 3, [1, 3, 3]),
        ([[0, 2, 3], [2, 2, 3]], 1, [[0, 1, 3], [2, 1, 3]]),
        ([[0, 2, 3], [2, 2, 3]], [-1, 1], [[0, -1, 3], [2, 1, 3]]),
    ],
    indirect=["vector"],
)
def test_assign_y(vector, data, expected):
    vector.y = data
    assert np.allclose(vector.data, expected)


@pytest.mark.parametrize(
    "vector, data, expected",
    [
        ([1, 2, 3], 1, [1, 2, 1]),
        ([[0, 2, 3], [2, 2, 3]], 1, [[0, 2, 1], [2, 2, 1]]),
        ([[0, 2, 3], [2, 2, 3]], [-1, 1], [[0, 2, -1], [2, 2, 1]]),
    ],
    indirect=["vector"],
)
def test_assign_z(vector, data, expected):
    vector.z = data
    assert np.allclose(vector.data, expected)


@pytest.mark.parametrize(
    "vector",
    [
        [(1, 0, 0)],
        [(0.5, 0.5, 1.25), (-1, -1, -1)],
    ],
    indirect=["vector"],
)
def test_perpendicular(vector: Vector3d):
    assert np.allclose(vector.dot(vector.perpendicular).data, 0)


def test_mean_xyz():
    x = Vector3d.xvector()
    y = Vector3d.yvector()
    z = Vector3d.zvector()
    t = Vector3d([3 * x.data, 3 * y.data, 3 * z.data])
    np.allclose(t.mean().data, 1)


@pytest.mark.xfail(strict=True, reason=ValueError)
def test_zero_perpendicular():
    t = Vector3d(np.asarray([0, 0, 0]))
    tperp = t.perpendicular()


@pytest.mark.xfail(strict=True, reason=TypeError)
class TestSpareNotImplemented:
    def test_radd_notimplemented(self, vector):
        "cantadd" + vector

    def test_rsub_notimplemented(self, vector):
        "cantsub" - vector

    def test_rmul_notimplemented(self, vector):
        "cantmul" * vector


class TestSphericalCoordinates:
    @pytest.mark.parametrize(
        "vector, theta_desired, phi_desired, r_desired",
        [
            (Vector3d((0.5, 0.5, 0.707107)), np.pi / 4, np.pi / 4, 1),
            (Vector3d((-0.75, -0.433013, -0.5)), 2 * np.pi / 3, 7 * np.pi / 6, 1),
        ],
    )
    def test_to_polar(self, vector, theta_desired, phi_desired, r_desired):
        theta, phi, r = vector.to_polar()
        assert np.allclose(theta.data, theta_desired)
        assert np.allclose(phi.data, phi_desired)
        assert np.allclose(r.data, r_desired)

    def test_polar_loop(self, vector):
        theta, phi, r = vector.to_polar()
        vector2 = Vector3d.from_polar(theta=theta.data, phi=phi.data, r=r.data)
        assert np.allclose(vector.data, vector2.data)
