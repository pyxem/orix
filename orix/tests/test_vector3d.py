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

from matplotlib.collections import QuadMesh
import matplotlib.pyplot as plt
import numpy as np
import pytest

from orix import plot
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d

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


def test_neg(vector):
    assert np.all((-vector).data == -(vector.data))


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        ([1, 2, 3], Vector3d([[1, 2, 3], [-3, -2, -1]]), [[2, 4, 6], [-2, 0, 2]]),
        ([1, 2, 3], [4], [5, 6, 7]),
        ([1, 2, 3], 0.5, [1.5, 2.5, 3.5]),
        ([1, 2, 3], [-1, 2], [[0, 1, 2], [3, 4, 5]]),
        ([1, 2, 3], np.array([-1, 1]), [[0, 1, 2], [2, 3, 4]]),
    ],
    indirect=["vector"],
)
def test_add(vector, other, expected):
    s1 = vector + other
    s2 = other + vector
    assert np.allclose(s1.data, expected)
    assert np.allclose(s1.data, s2.data)


@pytest.mark.parametrize(
    "vector, other",
    [([1, 2, 3], "dracula")],
    indirect=["vector"],
)
def test_add_raises(vector, other):
    with pytest.raises(TypeError):
        _ = vector + other


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        ([1, 2, 3], Vector3d([[1, 2, 3], [-3, -2, -1]]), [[0, 0, 0], [4, 4, 4]]),
        ([1, 2, 3], [4], [-3, -2, -1]),
        ([1, 2, 3], 0.5, [0.5, 1.5, 2.5]),
        ([1, 2, 3], [-1, 2], [[2, 3, 4], [-1, 0, 1]]),
        ([1, 2, 3], np.array([-1, 1]), [[2, 3, 4], [0, 1, 2]]),
    ],
    indirect=["vector"],
)
def test_sub(vector, other, expected):
    s1 = vector - other
    s2 = other - vector
    assert np.allclose(s1.data, expected)
    assert np.allclose(-s1.data, s2.data)


@pytest.mark.parametrize(
    "vector, other",
    [
        ([1, 2, 3], "dracula"),
    ],
    indirect=["vector"],
)
def test_sub_raises(vector, other):
    with pytest.raises(TypeError):
        _ = vector - other


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        ([1, 2, 3], [4], [4, 8, 12]),
        ([1, 2, 3], 0.5, [0.5, 1.0, 1.5]),
        ([1, 2, 3], [-1, 2], [[-1, -2, -3], [2, 4, 6]]),
        ([1, 2, 3], np.array([-1, 1]), [[-1, -2, -3], [1, 2, 3]]),
    ],
    indirect=["vector"],
)
def test_mul(vector, other, expected):
    s1 = vector * other
    s2 = other * vector
    assert np.allclose(s1.data, expected)
    assert np.allclose(s1.data, s2.data)


@pytest.mark.parametrize(
    "vector, other, error_type",
    [
        ([1, 2, 3], Vector3d([[1, 2, 3], [-3, -2, -1]]), ValueError),
        ([1, 2, 3], "dracula", TypeError),
    ],
    indirect=["vector"],
)
def test_mul_raises(vector, other, error_type):
    with pytest.raises(error_type):
        _ = vector * other


@pytest.mark.parametrize(
    "vector, other, expected",
    [
        ([4, 8, 12], [4], [1, 2, 3]),
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
    ],
    indirect=["vector"],
)
def test_div(vector, other, expected):
    s1 = vector / other
    assert np.allclose(s1.data, expected)


@pytest.mark.parametrize(
    "vector, other, error_type",
    [
        ([1, 2, 3], Vector3d([[1, 2, 3], [-3, -2, -1]]), ValueError),
        ([1, 2, 3], "dracula", TypeError),
    ],
    indirect=["vector"],
)
def test_div_raises(vector, other, error_type):
    with pytest.raises(error_type):
        _ = vector / other


def test_rdiv_raises():
    with pytest.raises(ValueError):
        _ = 1 / Vector3d.xvector()


def test_dot(vector, something):
    assert np.allclose(vector.dot(vector), (vector.data**2).sum(axis=-1))
    assert np.allclose(vector.dot(something), something.dot(vector))


def test_dot_error(vector, number):
    with pytest.raises(ValueError):
        vector.dot(number)


def test_dot_outer(vector, something):
    d = vector.dot_outer(something)
    assert d.shape == vector.shape + something.shape
    for i in np.ndindex(vector.shape):
        for j in np.ndindex(something.shape):
            assert np.allclose(d[i + j], vector[i].dot(something[j]))
    d_lazy = vector.dot_outer(something, lazy=True)
    assert isinstance(d_lazy, np.ndarray)
    assert np.allclose(d, d_lazy)
    d_lazy_no_pb = vector.dot_outer(
        something, lazy=True, progressbar=False, chunk_size=25
    )
    assert np.allclose(d_lazy, d_lazy_no_pb)


def test_dot_outer_progressbar(vector, something, capsys):
    d = vector.dot_outer(something, lazy=True, progressbar=True)
    out, _ = capsys.readouterr()
    assert "Completed" in out
    assert d.shape == vector.shape + something.shape


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
    v1 = Vector3d.from_polar(azimuth, polar, radial)
    v2 = Vector3d.from_polar(
        np.rad2deg(azimuth), np.rad2deg(polar), radial, degrees=True
    )
    assert np.allclose(v1.data, expected.data, atol=1e-5)
    assert np.allclose(v2.data, v1.data, atol=1e-5)


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
    a1 = vector.angle_with(vector)
    assert np.allclose(a1, 0)

    a2 = vector.angle_with(something)
    assert np.all(a2 >= 0)
    assert np.all(a2 <= np.pi)

    a3 = vector.angle_with(something, degrees=True)
    assert np.allclose(np.rad2deg(a2), a3)


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
    assert np.allclose(vector.dot(vector.perpendicular), 0)


def test_mean_xyz():
    x = Vector3d.xvector()
    y = Vector3d.yvector()
    z = Vector3d.zvector()
    t = Vector3d([3 * x.data, 3 * y.data, 3 * z.data])
    np.allclose(t.mean().data, 1)


def test_transpose_1d():
    v1 = Vector3d.random(7)
    v2 = v1.transpose()

    assert np.allclose(v1.data, v2.data)


@pytest.mark.parametrize(
    "shape, expected_shape",
    [
        ((6, 4), (4, 6)),
        ((11, 5), (5, 11)),
    ],
)
def test_transpose_2d_data_shape(shape, expected_shape):
    v1 = Vector3d.random(shape)
    assert v1.transpose().shape == expected_shape


def test_transpose_3d_no_axes():
    v1 = Vector3d.random((5, 4, 2))
    with pytest.raises(ValueError, match="Axes must be defined for more than"):
        _ = v1.transpose()


def test_transpose_3d_wrong_number_of_axes():
    v1 = Vector3d.random((5, 4, 2))
    with pytest.raises(ValueError, match="Number of axes is ill-defined"):
        _ = v1.transpose(0, 2)


@pytest.mark.parametrize(
    "shape, expected_shape",
    [
        ((6, 4), (4, 6)),
        ((11, 5), (5, 11)),
    ],
)
def test_transpose_2d_shape(shape, expected_shape):
    v1 = Vector3d.random(shape)
    assert v1.transpose().shape == expected_shape


@pytest.mark.parametrize(
    "shape, expected_shape, axes",
    [((6, 4, 5), (4, 5, 6), (1, 2, 0)), ((6, 4, 5), (5, 4, 6), (2, 1, 0))],
)
def test_transpose_3d_data_shape(shape, expected_shape, axes):
    v1 = Vector3d.random(shape)
    assert v1.transpose(*axes).shape == expected_shape


@pytest.mark.parametrize(
    "shape, expected_shape, axes",
    [((6, 4, 5), (4, 5, 6), (1, 2, 0)), ((6, 4, 5), (5, 4, 6), (2, 1, 0))],
)
def test_transpose_3d_shape(shape, expected_shape, axes):
    v1 = Vector3d.random(shape)
    assert v1.transpose(*axes).shape == expected_shape


def test_zero_perpendicular():
    with pytest.raises(ValueError):
        _ = Vector3d.zero((1,)).perpendicular


def test_get_nearest():
    v_ref = Vector3d.zvector()
    v = Vector3d([[0, 0, 0.9], [0, 0, 0.8], [0, 0, 1.1]])
    v_nearest = v_ref.get_nearest(v)
    assert np.allclose(v_nearest.data, [0, 0, 0.9])

    with pytest.raises(AttributeError, match="`get_nearest` only works for "):
        v.get_nearest(v_ref)


class TestSpareNotImplemented:
    def test_radd_notimplemented(self, vector):
        with pytest.raises(TypeError):
            _ = "cantadd" + vector

    def test_rsub_notimplemented(self, vector):
        with pytest.raises(TypeError):
            _ = "cantsub" - vector

    def test_rmul_notimplemented(self, vector):
        with pytest.raises(TypeError):
            _ = "cantmul" * vector


class TestVector3dInversePoleDensityFunction:
    def test_ipdf_plot(self):
        v = Vector3d.random(1_000)
        fig = v.inverse_pole_density_function(
            symmetry=symmetry.Th,
            return_figure=True,
            show_hemisphere_label=True,
        )
        assert len(fig.axes) == 2  # plot and colorbar
        qm1 = [isinstance(c, QuadMesh) for c in fig.axes[0].collections]
        assert any(qm1)
        plt.close(fig)

    def test_ipdf_plot_hemisphere_raises(self):
        with pytest.raises(ValueError, match="Hemisphere must be either "):
            v = Vector3d.random(1_000)
            _ = v.inverse_pole_density_function(
                symmetry=symmetry.Th,
                return_figure=True,
                hemisphere="test",
            )


class TestVector3dPoleDensityFunction:
    def test_pdf_plot_colorbar(self):
        v = Vector3d.random(10_000)
        fig1 = v.pole_density_function(return_figure=True, colorbar=True)
        assert len(fig1.axes) == 2  # plot and colorbar
        qm1 = [isinstance(c, QuadMesh) for c in fig1.axes[0].collections]
        assert any(qm1)
        plt.close(fig1)

        fig2 = v.pole_density_function(return_figure=True, colorbar=False)
        assert len(fig2.axes) == 1  # just plot
        plt.close(fig2)

        fig3 = v.pole_density_function(
            return_figure=True, hemisphere="both", colorbar=False
        )
        assert len(fig3.axes) == 2
        assert fig3.axes[0].hemisphere == "upper"
        assert fig3.axes[1].hemisphere == "lower"
        plt.close(fig3)

        fig4 = v.pole_density_function(return_figure=True, hemisphere="both")
        assert len(fig4.axes) == 4
        plt.close(fig4)

    def test_pdf_plot_hemisphere(self):
        v = Vector3d.random(10_000)
        fig1 = v.pole_density_function(return_figure=True, hemisphere="upper")
        qm1 = [isinstance(c, QuadMesh) for c in fig1.axes[0].collections]
        assert any(qm1)
        qmesh1 = fig1.axes[0].collections[qm1.index(True)].get_array().data
        plt.close(fig1)

        fig2 = v.pole_density_function(return_figure=True, hemisphere="lower")
        qm2 = [isinstance(c, QuadMesh) for c in fig2.axes[0].collections]
        assert any(qm2)
        qmesh2 = fig2.axes[0].collections[qm2.index(True)].get_array().data
        plt.close(fig2)

        # test mesh not the same, sigma is different
        assert not np.allclose(qmesh1, qmesh2)

        fig3 = v.pole_density_function(
            return_figure=True, colorbar=False, hemisphere="both"
        )
        qm3_1 = [isinstance(c, QuadMesh) for c in fig3.axes[0].collections]
        assert any(qm3_1)
        qmesh3_1 = fig3.axes[0].collections[qm3_1.index(True)].get_array().data
        qm3_2 = [isinstance(c, QuadMesh) for c in fig3.axes[1].collections]
        assert any(qm3_2)
        qmesh3_2 = fig3.axes[1].collections[qm3_2.index(True)].get_array().data
        plt.close(fig3)

        # test mesh the same as single plots
        assert np.allclose(qmesh1, qmesh3_1)  # upper
        assert np.allclose(qmesh2, qmesh3_2)  # lower

    def test_pdf_plot_sigma(self):
        v = Vector3d.random(10_000)
        fig1 = v.pole_density_function(return_figure=True)
        qm1 = [isinstance(c, QuadMesh) for c in fig1.axes[0].collections]
        assert any(qm1)
        qmesh1 = fig1.axes[0].collections[qm1.index(True)].get_array().data
        plt.close(fig1)

        fig2 = v.pole_density_function(return_figure=True, sigma=2)
        qm2 = [isinstance(c, QuadMesh) for c in fig2.axes[0].collections]
        assert any(qm2)
        qmesh2 = fig2.axes[0].collections[qm2.index(True)].get_array().data
        plt.close(fig2)

        # test mesh not the same, sigma is different
        assert not np.allclose(qmesh1, qmesh2)

    def test_pdf_plot_log(self):
        v = Vector3d.random(10_000)
        fig1 = v.pole_density_function(return_figure=True)
        qm1 = [isinstance(c, QuadMesh) for c in fig1.axes[0].collections]
        assert any(qm1)
        qmesh1 = fig1.axes[0].collections[qm1.index(True)].get_array().data
        plt.close(fig1)

        fig2 = v.pole_density_function(return_figure=True, log=True)
        qm2 = [isinstance(c, QuadMesh) for c in fig2.axes[0].collections]
        assert any(qm2)
        qmesh2 = fig2.axes[0].collections[qm2.index(True)].get_array().data
        plt.close(fig2)

        # test mesh not the same, log is different
        assert not np.allclose(qmesh1, qmesh2)

    def test_pdf_hemisphere_raises(self):
        v = Vector3d.random(100)
        with pytest.raises(ValueError, match=r"Hemisphere must be either "):
            _ = v.pole_density_function(return_figure=True, hemisphere="test")


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

        azimuth2, polar2, radial2 = v.to_polar(degrees=True)
        assert np.allclose(polar2.data, np.rad2deg(polar_desired))
        assert np.allclose(azimuth2.data, np.rad2deg(azimuth_desired))
        assert np.allclose(radial2.data, radial_desired)

    def test_polar_loop(self, vector):
        azimuth, polar, radial = vector.to_polar()
        vector2 = Vector3d.from_polar(azimuth=azimuth, polar=polar, radial=radial)
        assert np.allclose(vector.data, vector2.data)


class TestGetCircle:
    def test_get_circle(self):
        v = Vector3d([0, 0, 1])
        oa = 0.5 * np.pi
        c = v.get_circle(opening_angle=oa, steps=101)

        assert c.size == 101
        assert np.allclose(c.z, 0)
        assert np.allclose(v.angle_with(c), oa)
        assert np.allclose(c.mean().data, [0, 0, 0], atol=1e-2)
        assert np.allclose(v.cross(c[0, 0]).data, [1, 0, 0])

    def test_from_path_ends(self):
        vx = Vector3d.xvector()
        vy = Vector3d.yvector()

        v_xy = Vector3d.from_path_ends(Vector3d.stack((vx, vy)))
        assert v_xy.size == 27
        assert np.allclose(v_xy.polar, np.pi / 2)
        assert np.allclose(v_xy[-1].data, vy.data)

        v_xyz = Vector3d.from_path_ends(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], steps=150, close=True
        )
        assert v_xyz.size == 115
        assert np.allclose(v_xyz[-1].data, vx.data)

        with pytest.raises(ValueError, match="No vectors are perpendicular"):
            _ = Vector3d.from_path_ends(Vector3d.stack((vx, -vx)))


class TestPlotting:
    v = Vector3d(
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 0, -1], [-1, 0, -1], [-1, -1, -1]]
    )

    def test_scatter(self):
        """Test almost everything about the convenience method for
        scatter plots of 3D vectors.
        """
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
        labels = plot.format_labels(v.data)
        offset = (-0.02, 0.05)
        fig2 = v.scatter(
            hemisphere="both",
            grid=True,
            grid_resolution=(azimuth_res, polar_res),
            vector_labels=labels,
            figure_kwargs=dict(figsize=fig_size),
            text_kwargs=dict(size=text_size, offset=offset),
            c=scatter_colors,
            return_figure=True,
        )
        assert fig2 != fig1  # New figure
        assert fig2.get_figwidth() == fig_size[0]
        assert fig2.get_figheight() == fig_size[1]
        ax2 = fig2.axes
        assert len(ax2) == 2
        assert ax2[0]._azimuth_resolution == azimuth_res
        assert ax2[1]._polar_resolution == polar_res
        assert ax2[0].texts[0].get_text() == "upper"
        assert ax2[1].texts[0].get_text() == "lower"
        assert ax2[0].texts[1].get_text() == labels[0]
        assert ax2[1].texts[1].get_size() == text_size

        # Given offset in text_kwargs propagated to
        # StereographicPlot.text()
        text0 = ax2[0].texts[1]
        x, y = ax2[0]._projection.vector2xy(v[0])
        assert np.isclose(text0._x, x + offset[0])
        assert np.isclose(text0._y, y + offset[1])

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
        with pytest.raises(NotImplementedError, match="Projection "):
            self.v.scatter(projection="equal_angle")

    def test_scatter_reproject(self):
        o = Orientation.from_axes_angles((-1, 8, 1), 65, degrees=True)
        v = (symmetry.Oh * o) * Vector3d.zvector()

        # Normal scatter: half of the vectors are shown
        fig1 = v.scatter(return_figure=True)
        assert fig1.axes[0].collections[0].get_offsets().shape == (v.size // 2, 2)

        # Reproject: all of the vectors are shown
        fig2 = v.scatter(reproject=True, return_figure=True, c="r")
        colls = fig2.axes[0].collections[:2]
        for coll in colls[:2]:
            assert coll.get_offsets().shape == (v.size // 2, 2)
            assert np.allclose(coll.get_edgecolor(), (1, 0, 0, 1))

        # Reproject: all of the vectors are shown
        fig3 = v.scatter(hemisphere="lower", reproject=True, return_figure=True)
        colls = fig3.axes[0].collections[:2]
        for coll in colls:
            assert coll.get_offsets().shape == (v.size // 2, 2)

        # Reproject hemisphere="both": reprojection is ignored so half
        # of the vectors are shown on each axes as normal
        fig4 = v.scatter(hemisphere="both", reproject=True, return_figure=True)
        for ax in fig4.axes:
            assert ax.collections[0].get_offsets().shape == (v.size // 2, 2)

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

    def test_draw_circle_reproject(self):
        """Ensure that parts of circles on the other hemisphere are
        added with correct appearance to `StereographicPlot.lines`.
        """
        v = Vector3d([(1, 1, 1), (-1, 1, 1)])

        fig1 = v.draw_circle(return_figure=True)
        assert len(fig1.axes[0].lines) == 2

        fig2 = v.draw_circle(reproject=True, return_figure=True, linewidth=2)
        circles2 = fig2.axes[0].lines
        assert len(circles2) == 4
        for c, style in zip(circles2, ["-", "-", "--", "--"]):
            assert c.get_linestyle() == style
            assert c.get_color() == "C0"
            assert c.get_linewidth() == 2

        fig3 = v.draw_circle(
            reproject=True,
            return_figure=True,
            color=["r", "g"],
            reproject_plot_kwargs=dict(linestyle="-."),
        )
        circles3 = fig3.axes[0].lines
        assert len(circles3) == 4
        for c, style, color in zip(
            circles3, ["-", "-", "-.", "-."], ["r", "g", "r", "g"]
        ):
            assert c.get_linestyle() == style
            assert c.get_color() == color

        # Cover line where `reproject_plot_kwargs` is not passed
        # directly to `StereographicPlot.draw_circle()`
        fig4, ax4 = plt.subplots(subplot_kw=dict(projection="stereographic"))
        ax4.draw_circle(v, reproject=True)
        circles4 = ax4.lines
        assert len(circles4) == 4
        assert all([c.get_color() == "C0" for c in circles4])


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
