import pytest
import numpy as np

from orix.base import DimensionError, Object3d, check


@pytest.mark.xfail(strict=True, reason=ValueError)
def test_check_failing():
    check(np.asarray([1, 1, 1, 1]), Object3d)


@pytest.fixture(
    params=[
        (2,),
        (2, 2),
        (4, 2),
        (2, 3, 2),
        (100, 100, 2),
        (3,),
        (3, 3),
        (4, 3),
        (2, 3, 3),
        (100, 100, 3),
        (4,),
        (3, 4),
        (3, 4),
        (2, 3, 4),
        (100, 100, 4),
    ]
)
def data(request):
    np.random.seed(4)
    dat = np.random.rand(*request.param) * 2 - 1
    return dat


# Create an abstract subclass to test methods
@pytest.fixture(params=[2, 3, 4])
def test_object3d(request):
    class TestObject3d(Object3d):
        dim = request.param

    return TestObject3d


@pytest.fixture(
    params=[
        (1,),
        (2,),
        (3,),
        (4,),
        (8,),
        (100,),
        (1, 1,),
        (2, 1),
        (1, 2),
        (2, 2,),
        (5, 5,),
        (100, 100,),
        (1, 1, 1),
        (2, 1, 1),
        (1, 2, 1),
        (1, 4, 3),
        (6, 4, 3),
        (50, 40, 30),
    ]
)
def object3d(request, test_object3d):
    shape = request.param + (test_object3d.dim,)
    np.random.seed(4)
    dat = np.random.rand(*shape) * 2 - 1
    return test_object3d(dat)


@pytest.mark.parametrize(
    "test_object3d, data",
    [
        (2, (2,)),
        (2, (3, 2,)),
        (2, (4, 3, 2,)),
        (2, (5, 4, 3, 2,)),
        pytest.param(2, (3,), marks=pytest.mark.xfail(raises=DimensionError)),
        pytest.param(2, (2, 1), marks=pytest.mark.xfail(raises=DimensionError)),
        pytest.param(2, (3, 3), marks=pytest.mark.xfail(raises=DimensionError)),
        pytest.param(2, (3, 3, 3), marks=pytest.mark.xfail(raises=DimensionError)),
        (3, (3,)),
        (3, (4, 3,)),
        (3, (5, 4, 3,)),
        (3, (2, 5, 4, 3,)),
        pytest.param(3, (2,), marks=pytest.mark.xfail(raises=DimensionError)),
        pytest.param(3, (3, 1,), marks=pytest.mark.xfail(raises=DimensionError)),
        pytest.param(3, (2, 2,), marks=pytest.mark.xfail(raises=DimensionError)),
        pytest.param(3, (2, 2, 4), marks=pytest.mark.xfail(raises=DimensionError)),
    ],
    indirect=["test_object3d", "data"],
)
def test_init(test_object3d, data):
    obj = test_object3d(data)
    assert np.allclose(obj.data, data)


@pytest.mark.parametrize(
    "test_object3d, data, key",
    [
        (2, (5, 5, 2), (slice(0), slice(0))),
        (2, (5, 5, 2), slice(1)),
        (2, (5, 2), slice(1)),
        (2, (5, 5, 2), 3),
        (2, (5, 5, 2), slice(0, 3)),
        (2, (5, 5, 2), (None, slice(1, 5))),
        pytest.param(
            2,
            (5, 2),
            (slice(1), slice(1), slice(1),),
            marks=pytest.mark.xfail(raises=IndexError),
        ),
        pytest.param(2, (5, 2), slice(7, 8)),
        pytest.param(
            3,
            (4, 4, 3),
            (6, 6),
            marks=pytest.mark.xfail(raises=IndexError, strict=True),
        ),
    ],
    indirect=["test_object3d", "data"],
)
def test_slice(test_object3d, data, key):
    obj = test_object3d(data)
    obj_subset = obj[key]
    print(key)
    assert isinstance(obj_subset, test_object3d)
    assert np.allclose(obj_subset.data, data[key])


def test_shape(object3d):
    assert object3d.shape == object3d.data.shape[:-1]


def test_data_dim(object3d):
    assert object3d.data_dim == len(object3d.data.shape[:-1])


def test_size(object3d):
    assert object3d.size == object3d.data.size / object3d.dim


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_stack(object3d, n):
    stack = object3d.stack([object3d] * n)
    assert isinstance(stack, object3d.__class__)
    assert stack.shape[-1] == n


def test_flatten(object3d):
    flat = object3d.flatten()
    assert isinstance(flat, object3d.__class__)
    assert flat.data_dim == 1
    assert flat.shape[0] == object3d.size


@pytest.mark.parametrize("test_object3d", [1,], indirect=["test_object3d"])
def test_unique(test_object3d):
    object = test_object3d([[1], [1], [2], [3], [3]])
    unique = object.unique()
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    unique, idx = object.unique(return_index=True)
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    assert np.allclose(idx, [0, 2, 3])
    unique, inv = object.unique(return_inverse=True)
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    assert np.allclose(inv, [0, 0, 1, 2, 2])
    unique, idx, inv = object.unique(True, True)
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    assert np.allclose(idx, [0, 2, 3])
    assert np.allclose(inv, [0, 0, 1, 2, 2])
