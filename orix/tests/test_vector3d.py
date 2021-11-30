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

from orix.quaternion import symmetry
from orix.scalar import Scalar
from orix.vector import Vector3d, check_vector


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
    "azimuth, polar, radial, expected",
    [
        (np.pi / 4, np.pi / 4, 1, Vector3d((0.5, 0.5, 0.707107))),
        (7 * np.pi / 6, 2 * np.pi / 3, 1, Vector3d((-0.75, -0.433013, -0.5))),
    ],
)
def test_polar(azimuth, polar, radial, expected):
    assert np.allclose(
        Vector3d.from_polar(azimuth=azimuth, polar=polar, radial=radial).data,
        expected.data,
        atol=1e-5,
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
        ((1, 0, 0), np.pi / 2, (0, 1, 0)),
        ((1, 1, 0), np.pi / 2, (-1, 1, 0)),
        (
            (1, 1, 0),
            [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
            [(-1, 1, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0)],
        ),
        ((1, 1, 1), -np.pi / 2, (1, -1, 1)),
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


def test_transpose_1d():
    v1 = Vector3d(np.random.rand(7, 3))
    v2 = v1.transpose()

    assert np.allclose(v1.data, v2.data)


@pytest.mark.parametrize(
    "shape, expected_shape",
    [
        ([6, 4, 3], [4, 6, 3]),
        ([11, 5, 3], [5, 11, 3]),
    ],
)
def test_transpose_2d_data_shape(shape, expected_shape):
    v1 = Vector3d(np.random.rand(*shape))
    v2 = v1.transpose()

    assert v2.data.shape == tuple(expected_shape)


def test_transpose_3d_no_axes():
    v1 = Vector3d(np.random.rand(5, 4, 2, 3))
    with pytest.raises(ValueError, match="Axes must be defined for more than"):
        _ = v1.transpose()


def test_transpose_3d_wrong_number_of_axes():
    v1 = Vector3d(np.random.rand(5, 4, 2, 3))
    with pytest.raises(ValueError, match="Number of axes is ill-defined"):
        _ = v1.transpose(0, 2)


@pytest.mark.parametrize(
    "shape, expected_shape",
    [
        ([6, 4], [4, 6]),
        ([11, 5], [5, 11]),
    ],
)
def test_transpose_2d_shape(shape, expected_shape):
    v1 = Vector3d(np.random.rand(*shape, 3))
    v2 = v1.transpose()

    assert v2.shape == tuple(expected_shape)


@pytest.mark.parametrize(
    "shape, expected_shape, axes",
    [([6, 4, 5, 3], [4, 5, 6, 3], [1, 2, 0]), ([6, 4, 5, 3], [5, 4, 6, 3], [2, 1, 0])],
)
def test_transpose_3d_data_shape(shape, expected_shape, axes):
    v1 = Vector3d(np.random.rand(*shape))
    v2 = v1.transpose(*axes)

    assert v2.data.shape == tuple(expected_shape)


@pytest.mark.parametrize(
    "shape, expected_shape, axes",
    [([6, 4, 5], [4, 5, 6], [1, 2, 0]), ([6, 4, 5], [5, 4, 6], [2, 1, 0])],
)
def test_transpose_3d_shape(shape, expected_shape, axes):
    v1 = Vector3d(np.random.rand(*shape, 3))
    v2 = v1.transpose(*axes)

    assert v2.shape == tuple(expected_shape)


@pytest.mark.xfail(strict=True, reason=ValueError)
def test_zero_perpendicular():
    t = Vector3d(np.asarray([0, 0, 0]))
    _ = t.perpendicular()


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
        "v, polar_desired, azimuth_desired, radial_desired",
        [
            (Vector3d((0.5, 0.5, 0.707107)), np.pi / 4, np.pi / 4, 1),
            (Vector3d((-0.75, -0.433013, -0.5)), 2 * np.pi / 3, 7 * np.pi / 6, 1),
        ],
    )
    def test_to_polar(self, v, polar_desired, azimuth_desired, radial_desired):
        azimuth, polar, radial = v.to_polar()
        assert np.allclose(polar.data, polar_desired)
        assert np.allclose(azimuth.data, azimuth_desired)
        assert np.allclose(radial.data, radial_desired)

    def test_polar_loop(self, vector):
        azimuth, polar, radial = vector.to_polar()
        vector2 = Vector3d.from_polar(
            azimuth=azimuth.data, polar=polar.data, radial=radial.data
        )
        assert np.allclose(vector.data, vector2.data)


class TestGetCircle:
    def test_get_circle(self):
        v = Vector3d([0, 0, 1])
        oa = 0.5 * np.pi
        c = v.get_circle(opening_angle=oa, steps=101)

        assert c.size == 101
        assert np.allclose(c.z.data, 0)
        assert np.allclose(v.angle_with(c).data, oa)
        assert np.allclose(c.mean().data, [0, 0, 0], atol=1e-2)
        assert np.allclose(v.cross(c[0, 0]).data, [1, 0, 0])


class TestPlotting:
    v = Vector3d(
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, -1], [-1, 0, -1], [-1, -1, -1]]
    )

    def test_scatter(self):
        plt.rcParams["axes.grid"] = False
        v = self.v

        axes_labels = ["x", "y"]
        fig1 = v.scatter(return_figure=True, axes_labels=axes_labels)
        assert isinstance(fig1, plt.Figure)
        texts1 = fig1.axes[0].texts
        for i in range(len(axes_labels)):
            assert texts1[i].get_text() == axes_labels[i]

        azimuth_res = 15
        polar_res = 20
        fig_size = (5, 10)
        text_size = 20
        scatter_colors = ["C0", "C1", "C2"] * 2
        vector_labels = [str(vi).replace(" ", "") for vi in v.data]
        fig2 = v.scatter(
            hemisphere="both",
            grid=True,
            grid_resolution=(azimuth_res, polar_res),
            vector_labels=vector_labels,
            figure_kwargs=dict(figsize=fig_size),
            text_kwargs=dict(size=text_size),
            c=scatter_colors,
            return_figure=True,
        )
        assert fig2 != fig1  # New figure
        assert fig2.get_figwidth() == fig_size[0]
        assert fig2.get_figheight() == fig_size[1]
        assert len(fig2.axes) == 2
        assert fig2.axes[0]._azimuth_resolution == azimuth_res
        assert fig2.axes[1]._polar_resolution == polar_res
        assert fig2.axes[0].texts[0].get_text() == "upper"
        assert fig2.axes[1].texts[0].get_text() == "lower"
        assert fig2.axes[0].texts[1].get_text() == vector_labels[0]
        assert fig2.axes[1].texts[1].get_size() == text_size

        fig3 = v.scatter(figure=fig1, return_figure=True)
        assert fig3 == fig1

        plt.close("all")

    def test_scatter_grid(self):
        plt.rcParams["axes.grid"] = True
        v = self.v
        fig = v.scatter(grid=False, return_figure=True)

        # Would think this attribute controlled whether a grid was
        # visible or not, but it doesn't seem like it. This should be
        # looked into again if this ever is untrue!
        assert fig.axes[0]._gridOn is True

        # Custom attribute
        assert fig.axes[0]._stereographic_grid is False

        # Grid remains off, respecting _stereographic_grid
        v.scatter(figure=fig)
        assert fig.axes[0]._stereographic_grid is False

        # Grid is turned on, respecting `grid`
        v.scatter(figure=fig, grid=True)
        assert fig.axes[0]._stereographic_grid is True

        # Grid remains on, respecting _stereographic_grid
        plt.rcParams["axes.grid"] = False
        v.scatter(figure=fig)
        assert fig.axes[0]._stereographic_grid is True

        # New figure, so _stereographic_grid should be as `axes.grid`
        fig2 = v.scatter(return_figure=True)
        assert fig2.axes[0]._stereographic_grid is False

        # Grid is turned on, respecting `grid`
        v.scatter(figure=fig2, grid=True)
        assert fig2.axes[0]._stereographic_grid is True

        plt.close("all")

    def test_scatter_projection(self):
        with pytest.raises(
            NotImplementedError, match="Stereographic is the only supported"
        ):
            self.v.scatter(projection="equal_angle")

    def test_draw_circle(self):
        v = self.v
        colors = [f"C{i}" for i in range(v.size)]
        steps = 200
        fig1 = v.draw_circle(
            steps=steps,
            hemisphere="both",
            return_figure=True,
            color=colors,
            linestyle="--",
        )

        assert isinstance(fig1, plt.Figure)
        assert len(fig1.axes) == 2
        assert all(a.hemisphere == h for a, h in zip(fig1.axes, ["upper", "lower"]))
        assert fig1.axes[0].lines[0]._path._vertices.shape == (steps, 2)

    def test_draw_circle_grid(self):
        plt.rcParams["axes.grid"] = True
        v = self.v
        fig = v.draw_circle(grid=False, return_figure=True)

        # Would think this attribute controlled whether a grid was
        # visible or not, but it doesn't seem like it. This should be
        # looked into again if this ever is untrue!
        assert fig.axes[0]._gridOn is True

        # Custom attribute
        assert fig.axes[0]._stereographic_grid is False

        v.draw_circle(figure=fig)
        assert fig.axes[0]._stereographic_grid is False

        v.draw_circle(figure=fig, grid=True)
        assert fig.axes[0]._stereographic_grid is True

        plt.close("all")


class TestProjectingToFundamentalSector:
    def test_in_fundamental_sector_oh(self):
        # First two outside, last barely within
        v1 = Vector3d(((1, 0, 0), (1, 1, 0), (1, 0.9, 1.1)))
        point_group = symmetry.Oh
        fs = point_group.fundamental_sector
        assert np.allclose((False, False, True), v1 < fs)
        v2 = v1.in_fundamental_sector(point_group)
        assert np.all(v2 <= fs)
        assert np.allclose(((0, 0, 1), (1, 0, 1), (1, 0.9, 1.1)), v2.data)

    def test_in_fundamental_sector_special(self):
        v1 = Vector3d(((0, -1.1, 0.1), (0.9, 1, -1), (1, 0.9, 1.1)))

        # Triclinic, which has a sector with no center. Nothing changes
        point_group = symmetry.C1
        assert np.allclose(v1.data, v1.in_fundamental_sector(point_group).data)

        # Tetragonal -4
        point_group = symmetry.S4
        fs = point_group.fundamental_sector
        assert np.allclose((False, False, True), v1 < fs)
        v2 = v1.in_fundamental_sector(point_group)
        assert np.all(v2 < fs)

        # Trigonal 321, 312, -3
        point_group = symmetry.S6
        fs = point_group.fundamental_sector
        assert np.allclose((False, False, True), v1 < fs)
        v3 = v1.in_fundamental_sector(point_group)
        assert np.all(v3 < fs)
