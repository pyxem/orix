import pytest
import numpy as np

from orix.scalar import Scalar
from orix.vector import Vector3d


@pytest.fixture(params=[(1,)])
def scalar(request):
    return Scalar(request.param)


@pytest.mark.parametrize(
    "data, expected",
    [
        ((5, 3), (5, 3)),
        ([[1], [2]], [[1], [2]]),
        (np.array([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]),
        (Scalar([-1, 1]), [-1, 1]),
    ],
)
def test_init(data, expected):
    scalar = Scalar(data)
    assert np.allclose(scalar.data, expected)


@pytest.mark.parametrize(
    "scalar, expected", [(1, -1), ((1, -1), (-1, 1))], indirect=["scalar"]
)
def test_neg(scalar, expected):
    neg = -scalar
    assert np.allclose(neg.data, expected)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        (1, 1, 2),
        ((1, 2), (2, -1), (3, 1)),
        ([[0, -1], [4, 2]], 0.5, [[0.5, -0.5], [4.5, 2.5]]),
        ((4,), np.array([[-1, -1], [-1, -1]]), [[3, 3], [3, 3]]),
        ((-1,), Scalar((1,)), [0]),
        pytest.param(0.5, "frederick", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_add(scalar, other, expected):
    sum = scalar + other
    assert np.allclose(sum.data, expected)
    sum2 = other + scalar
    assert np.allclose(sum.data, sum2.data)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        (1, 1, 0),
        ((1, 2), (2, -1), (-1, 3)),
        ([[0, -1], [4, 2]], 0.5, [[-0.5, -1.5], [3.5, 1.5]]),
        ((4,), np.array([[-1, -2], [1, -1]]), [[5, 6], [3, 5]]),
        ((-1,), Scalar((1,)), [-2]),
        pytest.param(0.5, "frederick", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_sub(scalar, other, expected):
    sub = scalar - other
    assert np.allclose(sub.data, expected)
    sub2 = other - scalar
    assert np.allclose(sub.data, -sub2.data)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        (1, 1, 1),
        ((1, 2), (2, -1), (2, -2)),
        ([[0, -1], [4, 2]], 0.5, [[0, -0.5], [2, 1]]),
        ((4,), np.array([[-1, -2], [1, -1]]), [[-4, -8], [4, -4]]),
        ((-1,), Scalar((1,)), [-1]),
        pytest.param(0.5, "frederick", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_mul(scalar, other, expected):
    mul = scalar * other
    assert np.allclose(mul.data, expected)
    mul2 = other * scalar
    assert np.allclose(mul.data, mul2.data)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        pytest.param(1, 1, 0, marks=pytest.mark.xfail),
        ((2, 2), (2, -1), (1, 0)),
        ([[0.5, -1], [4, 2]], 0.5, [[1, 0], [0, 0]]),
        ((4,), np.array([[-1, -2], [4, -1]]), [[0, 0], [1, 0]]),
        ([5], Scalar([5]), 1),
        ([5], Scalar([6]), 0),
        pytest.param([3], "larry", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_equality(scalar, other, expected):
    eq = scalar == other
    assert np.allclose(eq, expected)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        pytest.param(1, 1, 0, marks=pytest.mark.xfail),
        ((1, 2), (2, -1), (0, 1)),
        ([[0, -1], [4, 2]], 0.5, [[0, 0], [1, 1]]),
        ((4,), np.array([[-1, -2], [1, -1]]), [[1, 1], [1, 1]]),
        ([5], Scalar([6]), 0),
        pytest.param([3], "larry", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_inequality(scalar, other, expected):
    gt = scalar > other
    assert np.allclose(gt, expected)
    lt = scalar < other
    assert np.allclose(gt, ~lt)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        (1, 1, 1),
        ((1, 2), (2, -1), (0, 1)),
        ([[0, -1], [4, 2]], 0.5, [[0, 0], [1, 1]]),
        ((1,), np.array([[-1, -2], [1, -1]]), [[1, 1], [1, 1]]),
        ([5], Scalar([5]), 1),
        ([5], Scalar([6]), 0),
        pytest.param([3], "larry", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_ge(scalar, other, expected):
    gt = scalar >= other
    assert np.allclose(gt, expected)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        (1, 1, 1),
        ((1, 2), (2, -1), (1, 0)),
        ([[0, -1], [4, 2]], 0.5, [[1, 1], [0, 0]]),
        ((1,), np.array([[-1, -2], [1, -1]]), [[0, 0], [1, 0]]),
        ([5], Scalar([5]), 1),
        ([5], Scalar([6]), 1),
        pytest.param([3], "larry", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_le(scalar, other, expected):
    le = scalar <= other
    assert np.allclose(le, expected)


@pytest.mark.parametrize(
    "scalar, other, expected",
    [
        (1, 1, 1),
        ((1.0, 2.0), (2, -1), (1, 0.5)),
        ([[0, -1], [4, 2]], 2, [[0, 1], [16, 4]]),
        ((4.0,), np.array([[-1, -2], [1, -1]]), [[0.25, 0.0625], [4, 0.25]]),
        pytest.param([3], "larry", None, marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_pow(scalar, other, expected):
    pow = scalar ** other
    assert np.allclose(pow.data, expected)


@pytest.mark.parametrize(
    "scalar, expected",
    [((1, 1, 1), (3,)), ([[0, -1], [4, 2]], (2, 2)), ([[5, 1, 0]], (1, 3)),],
    indirect=["scalar"],
)
def test_shape(scalar, expected):
    shape = scalar.shape
    assert shape == expected


@pytest.mark.parametrize(
    "scalar, shape, expected",
    [
        ((1, 1, 1), (3, 1), np.array([[1], [1], [1]])),
        ([[0, -1], [4, 2]], (4,), [0, -1, 4, 2]),
        pytest.param([[0, -1], [4, 2]], (3,), [0, -1, 4], marks=pytest.mark.xfail),
    ],
    indirect=["scalar"],
)
def test_reshape(scalar, shape, expected):
    s = scalar.reshape(*shape)
    assert s.shape == shape
    assert np.allclose(s.data, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            [Scalar([[1, 2], [3, 4]]), Scalar([[5, 6], [7, 8]])],
            [[[1, 5], [2, 6]], [[3, 7], [4, 8]]],
        ),
    ],
)
def test_stack(data, expected):
    stack = Scalar.stack(data)
    assert isinstance(stack, Scalar)
    assert stack.shape[-1] == len(data)
    assert np.allclose(stack.data, expected)


def test_flatten(scalar):
    scalar.flatten()
    return None


@pytest.mark.xfail(strict=True, reason=TypeError)
class TestSpareNotImplemented:
    def test_radd_notimplemented(self, scalar):
        "cantadd" + scalar

    def test_rsub_notimplemented(self, scalar):
        "cantsub" - scalar

    def test_rmul_notimplemented(self, scalar):
        "cantmul" * scalar

    def test_lt_notimplemented(self, scalar):
        scalar < "cantlt"
