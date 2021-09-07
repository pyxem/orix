# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
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

import matplotlib.pyplot as plt
import numpy as np
import pytest

from orix.plot import AxAnglePlot, RodriguesPlot
from orix.quaternion import Misorientation, Orientation, Rotation
from orix.quaternion.symmetry import C1, C2, C3, C4, D2, D3, D6, T, O, Oh
from orix.scalar import Scalar
from orix.vector import AxAngle, Vector3d


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
    """Tests that the property Misorientation.equivalent runs without error,
    use grain_exchange=True as this falls back to grain_exchange=False when
    Gl!=Gr:

    Gl == C4 is grain exchange
    Gl == C2 is no grain exchange
    """
    m = Misorientation([1, 1, 1, 1])  # any will do
    m_new = m.set_symmetry(Gl, C4, verbose=True)
    m_new.symmetry
    _ = m_new.equivalent(grain_exchange=True)


def test_repr():
    m = Misorientation([1, 1, 1, 1])  # any will do
    _ = repr(m)


def test_repr_ori():
    shape = (2, 3)
    o = Orientation.identity(shape).set_symmetry(O)
    assert repr(o).split("\n")[0] == f"Orientation {shape} {O.name}"


def test_sub():
    o = Orientation([1, 1, 1, 1])  # any will do
    o = o.set_symmetry(C4)  # only one as it a O
    m = o - o
    assert np.allclose(m.data, [1, 0, 0, 0])


def test_sub_orientation_and_other():
    m = Orientation([1, 1, 1, 1])  # any will do
    with pytest.raises(TypeError):
        _ = m - 3


def test_transpose_2d():
    o1 = Orientation.random_vonmises((11, 3))
    o2 = o1.transpose()

    assert o1.shape == o2.shape[::-1]


@pytest.mark.parametrize(
    "shape, expected_shape, axes",
    [((11, 3, 5), (11, 5, 3), (0, 2, 1)), ((11, 3, 5), (3, 5, 11), (1, 2, 0))],
)
def test_transpose_3d(shape, expected_shape, axes):
    o1 = Orientation.random_vonmises(shape)
    o2 = o1.transpose(*axes)

    assert o2.shape == tuple(expected_shape)


def test_transpose_symmetry():
    o1 = Orientation.random_vonmises((11, 3)).set_symmetry(Oh)
    o2 = o1.transpose()

    assert o1.symmetry == o2.symmetry


class TestOrientationInitialization:
    def test_from_euler_symmetry(self):
        euler = np.deg2rad([90, 45, 90])
        o1 = Orientation.from_euler(euler)
        assert np.allclose(o1.data, [0, -0.3827, 0, -0.9239], atol=1e-4)
        assert o1.symmetry.name == "1"
        o2 = Orientation.from_euler(euler, symmetry=Oh)
        assert np.allclose(o2.data, [0.9239, 0, 0.3827, 0], atol=1e-4)
        assert o2.symmetry.name == "m-3m"
        o3 = o1.set_symmetry(Oh)
        assert np.allclose(o3.data, o2.data)

    def test_from_matrix_symmetry(self):
        om = np.array(
            [np.eye(3), np.eye(3), np.diag([1, -1, -1]), np.diag([1, -1, -1])]
        )
        o1 = Orientation.from_matrix(om)
        assert np.allclose(
            o1.data, np.array([1, 0, 0, 0] * 2 + [0, 1, 0, 0] * 2).reshape((4, 4))
        )
        assert o1.symmetry.name == "1"
        o2 = Orientation.from_matrix(om, symmetry=Oh)
        assert np.allclose(
            o2.data, np.array([1, 0, 0, 0] * 2 + [-1, 0, 0, 0] * 2).reshape((4, 4))
        )
        assert o2.symmetry.name == "m-3m"
        o3 = o1.set_symmetry(Oh)
        assert np.allclose(o3.data, o2.data)

    def test_from_neo_euler_symmetry(self):
        v = AxAngle.from_axes_angles(axes=Vector3d.zvector(), angles=np.pi / 2)
        o1 = Orientation.from_neo_euler(v)
        assert np.allclose(o1.data, [0.7071, 0, 0, 0.7071])
        assert o1.symmetry.name == "1"
        o2 = Orientation.from_neo_euler(v, symmetry=Oh)
        assert np.allclose(o2.data, [-1, 0, 0, 0])
        assert o2.symmetry.name == "m-3m"
        o3 = o1.set_symmetry(Oh)
        assert np.allclose(o3.data, o2.data)


class TestOrientation:
    @pytest.mark.parametrize("symmetry", [C1, C2, C3, C4, D2, D3, D6, T, O, Oh])
    def test_get_distance_matrix(self, symmetry):
        q = [(0.5, 0.5, 0.5, 0.5), (0.5 ** 0.5, 0, 0, 0.5 ** 0.5)]
        o = Orientation(q).set_symmetry(symmetry)
        angles_numpy = o.get_distance_matrix()
        assert isinstance(angles_numpy, Scalar)
        assert angles_numpy.shape == (2, 2)

        angles_dask = o.get_distance_matrix(lazy=True)
        assert isinstance(angles_dask, Scalar)
        assert angles_dask.shape == (2, 2)

        assert np.allclose(angles_numpy.data, angles_dask.data)

    def test_get_distance_matrix_lazy_parameters(self):
        shape = (5, 15, 4)
        rng = np.random.default_rng()
        abcd = rng.normal(size=np.prod(shape)).reshape(shape)
        o = Orientation(abcd)

        angle1 = o.get_distance_matrix(lazy=True, chunk_size=5, progressbar=True)
        angle2 = o.get_distance_matrix(lazy=True, chunk_size=10, progressbar=False)

        assert np.allclose(angle1.data, angle2.data)

    @pytest.mark.parametrize("symmetry", [C1, C2, C3, C4, D2, D3, D6, T, O, Oh])
    def test_angle_with(self, symmetry):
        q = [(0.5, 0.5, 0.5, 0.5), (0.5 ** 0.5, 0, 0, 0.5 ** 0.5)]
        r = Rotation(q)
        o = Orientation(q).set_symmetry(symmetry)

        is_equal = np.allclose((~o).angle_with(o).data, (~r).angle_with(r).data)
        if symmetry.name in ["1", "m3m"]:
            assert is_equal
        else:
            assert not is_equal

    def test_negate_orientation(self):
        o = Orientation.identity().set_symmetry(Oh)
        on = -o
        assert on.symmetry.name == o.symmetry.name

    @pytest.mark.parametrize("pure_misorientation", [True, False])
    def test_scatter(self, orientation, pure_misorientation):
        if pure_misorientation:
            orientation = Misorientation(orientation)
            orientation = orientation.set_symmetry(C2, D6)
        fig_axangle = orientation.scatter(return_figure=True)
        assert isinstance(fig_axangle.axes[0], AxAnglePlot)
        fig_rodrigues = orientation.scatter(projection="rodrigues", return_figure=True)
        assert isinstance(fig_rodrigues.axes[0], RodriguesPlot)

        # Add multiple axes to figure, one at a time
        fig_multiple = plt.figure(figsize=(10, 5))
        assert len(fig_multiple.axes) == 0
        orientation.scatter(figure=fig_multiple, position=(1, 2, 1))

        # Figure is updated inplace
        assert len(fig_multiple.axes) == 1

        orientation.scatter(
            figure=fig_multiple,
            position=122,
            projection="rodrigues",
            wireframe_kwargs=dict(color="black", rcount=180),
            s=50,
        )
        assert len(fig_multiple.axes) == 2

        assert isinstance(fig_multiple.axes[0], AxAnglePlot)
        assert isinstance(fig_multiple.axes[1], RodriguesPlot)

        # Allow plotting a sub sample of the orientations
        orientation.random_vonmises(200).scatter(size=50)

        plt.close("all")
