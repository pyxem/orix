# Copyright 2018-2024 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest

from orix._base import DimensionError, Object3d


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
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
        (5, 5),
        (100, 100),
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
        (2, (3, 2)),
        (2, (4, 3, 2)),
        (2, (5, 4, 3, 2)),
        (3, (3,)),
        (3, (4, 3)),
        (3, (5, 4, 3)),
        (3, (2, 5, 4, 3)),
    ],
    indirect=["test_object3d", "data"],
)
def test_init(test_object3d, data):
    obj = test_object3d(data)
    assert np.allclose(obj.data, data)


@pytest.mark.parametrize(
    "test_object3d, data",
    [
        (2, (3,)),
        (2, (2, 1)),
        (2, (3, 3)),
        (2, (3, 3, 3)),
        (3, (2,)),
        (3, (3, 1)),
        (3, (2, 2)),
        (3, (2, 2, 4)),
    ],
    indirect=["test_object3d", "data"],
)
def test_init_fails(test_object3d, data):
    with pytest.raises(DimensionError):
        _ = test_object3d(data)


@pytest.mark.parametrize(
    "test_object3d, data, key",
    [
        (2, (5, 5, 2), (slice(0), slice(0))),
        (2, (5, 5, 2), slice(1)),
        (2, (5, 2), slice(1)),
        (2, (5, 5, 2), 3),
        (2, (5, 5, 2), slice(0, 3)),
        (2, (5, 5, 2), (None, slice(1, 5))),
        (2, (5, 2), slice(7, 8)),
    ],
    indirect=["test_object3d", "data"],
)
def test_slice(test_object3d, data, key):
    obj = test_object3d(data)
    obj_subset = obj[key]
    assert isinstance(obj_subset, test_object3d)
    assert np.allclose(obj_subset.data, data[key])


@pytest.mark.parametrize(
    "test_object3d, data, key, error_type",
    [
        (2, (5, 2), (slice(1), slice(1), slice(1)), IndexError),
        (3, (4, 4, 3), (6, 6), IndexError),
    ],
    indirect=["test_object3d", "data"],
)
def test_slice_raises(test_object3d, data, key, error_type):
    obj = test_object3d(data)
    with pytest.raises(error_type):
        _ = obj[key]


def test_shape(object3d):
    assert object3d.shape == object3d.data.shape[:-1]


def test_ndim(object3d):
    assert object3d.ndim == len(object3d.data.shape[:-1])


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
    assert flat.ndim == 1
    assert flat.shape[0] == object3d.size


@pytest.mark.parametrize("test_object3d", [1], indirect=["test_object3d"])
def test_unique(test_object3d):
    o3d = test_object3d([[1], [1], [2], [3], [3]])
    unique = o3d.unique()
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    unique, idx = o3d.unique(return_index=True)
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    assert np.allclose(idx, [0, 2, 3])
    unique, inv = o3d.unique(return_inverse=True)
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    assert np.allclose(inv, [0, 0, 1, 2, 2])
    unique, idx, inv = o3d.unique(True, True)
    assert np.allclose(unique.data.flatten(), [1, 2, 3])
    assert np.allclose(idx, [0, 2, 3])
    assert np.allclose(inv, [0, 0, 1, 2, 2])


@pytest.mark.parametrize("test_object3d", [4], indirect=["test_object3d"])
def test_get_random_sample(test_object3d):
    o3d = test_object3d(np.arange(80).reshape(5, 4, 4))
    o3d_sample = o3d.get_random_sample(10)
    assert o3d_sample.size == 10

    with pytest.raises(ValueError, match="Cannot draw a sample greater than 20"):
        _ = o3d.get_random_sample(21)
