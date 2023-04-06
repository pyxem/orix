# -*- coding: utf-8 -*-
# Copyright 2018-2023 the orix developers
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

import warnings

from diffpy.structure import Lattice, Structure
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciPyRotation

# fmt: off
# isort: off
from orix.crystal_map import Phase
from orix.plot import AxAnglePlot, InversePoleFigurePlot, RodriguesPlot
from orix.quaternion import Misorientation, Orientation, Rotation
from orix.quaternion.symmetry import (
    C1,
    C2,
    D2,
    C4,
    C3,
    D3,
    D6,
    T,
    O,
    Oh,
    _proper_groups,
)
from orix.vector import AxAngle, Miller, Vector3d 
# isort: on
# fmt: on


@pytest.fixture
def vector(request):
    return Vector3d(request.param)


@pytest.fixture(params=[(0.5, 0.5, 0.5, 0.5), (0.5**0.5, 0, 0, 0.5**0.5)])
def orientation(request):
    return Orientation(request.param)


class TestOrientation:
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
    def test_set_symmetry(self, orientation, symmetry, expected):
        o = Orientation(orientation.data, symmetry=symmetry)
        o = o.reduce()
        assert np.allclose(o.data, expected, atol=1e-3)

    @pytest.mark.parametrize(
        "symmetry, vector",
        [(C1, (1, 2, 3)), (C2, (1, -1, 3)), (C3, (1, 1, 1)), (O, (0, 1, 0))],
        indirect=["vector"],
    )
    def test_rotate_vector_after_reduction(self, symmetry, vector):
        """Ensure that a set of vectors rotated before and after
        reduction of an orientation to the Rodrigues fundamental zone
        are equivalent in the orientation's fundamental sector on S2.
        """
        v = symmetry.outer(vector).flatten()
        o = Orientation.random()
        oc = Orientation(o.data, symmetry=symmetry)
        oc = oc.reduce()
        v1 = o * v
        v1r = v1.in_fundamental_sector(oc.symmetry)
        v2 = oc * v
        v2r = v2.in_fundamental_sector(oc.symmetry)
        assert np.allclose(v1r.data, v2r.data, atol=1e-4)

    @pytest.mark.parametrize("symmetry", [C1, C2, C4, D2, D6, T, O])
    def test_getitem(self, orientation, symmetry):
        orientation.symmetry = symmetry
        assert orientation[0].symmetry._tuples == symmetry._tuples

    @pytest.mark.parametrize("symmetry", [C1, C2, C4, D2, D6, T, O])
    def test_reshape_maintains_symmetry(self, symmetry):
        o = Orientation.random((4, 5))
        o.symmetry = symmetry
        o1 = o.reshape(5, 4)
        assert o1.symmetry._tuples == symmetry._tuples

    @pytest.mark.parametrize("symmetry", [C1, C2, C4, D2, D6, T, O])
    def test_transpose_maintains_symmetry(self, symmetry):
        o = Orientation.random((4, 5))
        o.symmetry = symmetry
        o1 = o.transpose()
        assert o1.symmetry._tuples == symmetry._tuples

    @pytest.mark.parametrize("symmetry", [C1, C2, C4, D2, D6, T, O])
    def test_flatten_maintains_symmetry(self, symmetry):
        o = Orientation.random((4, 5))
        o.symmetry = symmetry
        o1 = o.flatten()
        assert o1.symmetry._tuples == symmetry._tuples

    @pytest.mark.parametrize("symmetry", [C1, C2, C4, D2, D6, T, O])
    def test_squeeze_maintains_symmetry(self, symmetry):
        o = Orientation.random((4, 5))
        o.symmetry = symmetry
        o1 = o.squeeze()
        assert o1.symmetry._tuples == symmetry._tuples

    def test_string_representation(self):
        shape = (2, 3)
        o = Orientation.identity(shape)
        o.symmetry = O
        o = o.reduce()
        assert repr(o).split("\n")[0] == f"Orientation {shape} {O.name}"

    def test_subtract(self):
        o = Orientation([1, 1, 1, 1], symmetry=C4)  # any will do
        o = o.reduce()
        m = o - o
        assert np.allclose(m.data, [1, 0, 0, 0])

    def test_subtraction_error(self):
        o = Orientation([1, 1, 1, 1])  # any will do
        with pytest.raises(TypeError):
            _ = o - 3

    def test_transpose_2d(self):
        o1 = Orientation.random_vonmises((11, 3))
        o2 = o1.transpose()
        assert o1.shape == o2.shape[::-1]

    def test_map_into_reduced_symmetry_zone_verbose(self):
        o = Orientation.random()
        o.symmetry = Oh
        with pytest.warns(np.VisibleDeprecationWarning):
            o1 = o.map_into_symmetry_reduced_zone()
            o2 = o.map_into_symmetry_reduced_zone(verbose=True)
        assert np.allclose(o1.data, o2.data)

    @pytest.mark.parametrize(
        "shape, expected_shape, axes",
        [((11, 3, 5), (11, 5, 3), (0, 2, 1)), ((11, 3, 5), (3, 5, 11), (1, 2, 0))],
    )
    def test_transpose_3d(self, shape, expected_shape, axes):
        o1 = Orientation.random_vonmises(shape)
        o2 = o1.transpose(*axes)
        assert o2.shape == tuple(expected_shape)

    def test_transpose_symmetry(self):
        o1 = Orientation.random_vonmises((11, 3))
        o1.symmetry = Oh
        o1 = o1.reduce()
        o2 = o1.transpose()
        assert o1.symmetry == o2.symmetry

    def test_symmetry_property(self):
        o = Orientation.random((3, 2))
        sym = Oh
        o.symmetry = sym
        assert o.symmetry == sym
        assert o._symmetry == (C1, sym)

    def test_symmetry_property_data(self):
        """Test that data remains unchanged after setting symmetry property."""
        o = Orientation.random((3, 2))
        d1 = o.data.copy()
        o.symmetry = Oh
        assert np.allclose(o.data, d1)

    def test_symmetry_property_wrong_type_orientation(self):
        o = Orientation.random((3, 2))
        with pytest.raises(TypeError, match="Value must be an instance of"):
            o.symmetry = 1

    def test_equality(self):
        # symmetries must also be the same to be equal
        o1 = Orientation.random((6, 5))
        o2 = Orientation(o1)
        assert o1 == o2
        o1.symmetry = C4
        o2.symmetry = o1.symmetry
        assert o1 == o2
        o2.symmetry = C3
        assert o1 != o2
        o3 = Orientation.random((6,))
        assert o1 != o3
        o3.symmetry = o1.symmetry
        assert o1 != o3

    @pytest.mark.parametrize("symmetry", [C1, C2, C3, C4, D2, D3, D6, T, O, Oh])
    def test_get_distance_matrix(self, symmetry):
        q = [(0.5, 0.5, 0.5, 0.5), (0.5**0.5, 0, 0, 0.5**0.5)]
        o = Orientation(q, symmetry=symmetry)
        o = o.reduce()
        angles_numpy = o.get_distance_matrix()
        assert isinstance(angles_numpy, np.ndarray)
        assert angles_numpy.shape == (2, 2)

        angles_dask = o.get_distance_matrix(lazy=True)
        assert isinstance(angles_dask, np.ndarray)
        assert angles_dask.shape == (2, 2)

        assert np.allclose(angles_numpy, angles_dask)
        assert np.allclose(np.diag(angles_numpy), 0)

        angles3 = o.get_distance_matrix(degrees=True)
        assert np.allclose(np.rad2deg(angles_numpy), angles3)

    def test_get_distance_matrix_lazy_parameters(self):
        shape = (5, 15, 4)
        rng = np.random.default_rng()
        abcd = rng.normal(size=np.prod(shape)).reshape(shape)
        o = Orientation(abcd)

        angle1 = o.get_distance_matrix(lazy=True, chunk_size=5)
        angle2 = o.get_distance_matrix(lazy=True, chunk_size=10, progressbar=False)

        assert np.allclose(angle1.data, angle2.data)

    @pytest.mark.parametrize("symmetry", [C1, C2, C3, C4, D2, D3, D6, T, O, Oh])
    def test_angle_with_outer(self, symmetry):
        shape = (5,)
        o = Orientation.random(shape)
        o.symmetry = symmetry

        dist = o.get_distance_matrix()
        awo_self = o.angle_with_outer(o)
        assert awo_self.shape == shape + shape
        assert np.allclose(dist, awo_self)
        assert np.allclose(np.diag(awo_self), 0, atol=1e-6)

        o2 = Orientation.random((6,))
        dist = o.angle_with_outer(o2)
        assert dist.shape == o.shape + o2.shape

        dist2 = o2.angle_with_outer(o)
        assert dist2.shape == o2.shape + o.shape

        assert np.allclose(dist, dist2.T)

    def test_angle_with_outer_shape(self):
        shape1 = (6, 5)
        shape2 = (7, 2)
        o1 = Orientation.random(shape1)
        o2 = Orientation.random(shape2)

        awo_o12 = o1.angle_with_outer(o2)
        assert awo_o12.shape == shape1 + shape2

        awo_o21 = o2.angle_with_outer(o1)
        assert awo_o21.shape == shape2 + shape1

        r1 = Rotation(o1)
        r2 = Rotation(o2)
        awo_r12 = r1.angle_with_outer(r2)
        awo_r21 = r2.angle_with_outer(r1)
        assert awo_o12.shape == awo_r12.shape
        assert awo_o21.shape == awo_r21.shape
        assert np.allclose(awo_o12, awo_r12)
        assert np.allclose(awo_o21, awo_r21)

        o1.symmetry = Oh
        awo_o12s = o1.angle_with_outer(o2)
        assert awo_o12s.shape == awo_r12.shape
        assert not np.allclose(awo_o12s, awo_r12)

    @pytest.mark.parametrize("symmetry", [C1, C2, C3, C4, D2, D3, D6, T, O, Oh])
    def test_angle_with(self, symmetry):
        q = [(0.5, 0.5, 0.5, 0.5), (0.5**0.5, 0, 0, 0.5**0.5)]
        r = Rotation(q)
        o = Orientation(q, symmetry=symmetry)

        is_equal = np.allclose((~o).angle_with(o), (~r).angle_with(r))
        if symmetry.name in ["1", "m3m"]:
            assert is_equal
        else:
            assert not is_equal

        ang_rad = (~o).angle_with(o)
        ang_deg = (~o).angle_with(o, degrees=True)
        assert np.allclose(np.rad2deg(ang_rad), ang_deg)

    def test_negate_orientation(self):
        o = Orientation.identity()
        o.symmetry = Oh
        o = o.reduce()
        on = -o
        assert on.symmetry.name == o.symmetry.name


class TestOrientationPlotting:
    @pytest.mark.parametrize("pure_misorientation", [True, False])
    def test_scatter(self, orientation, pure_misorientation):
        if pure_misorientation:
            orientation = Misorientation(orientation)
            orientation.symmetry = (C2, D6)
            orientation = orientation.reduce()
        fig_size = (5, 5)
        fig_axangle = orientation.scatter(
            return_figure=True, figure_kwargs=dict(figsize=fig_size)
        )
        assert (fig_axangle.get_size_inches() == fig_size).all()
        assert isinstance(fig_axangle.axes[0], AxAnglePlot)
        fig_rodrigues = orientation.scatter(projection="rodrigues", return_figure=True)
        assert isinstance(fig_rodrigues.axes[0], RodriguesPlot)

        # Add multiple axes to figure, one at a time
        fig_multiple = plt.figure(figsize=(10, 5))
        assert len(fig_multiple.axes) == 0
        orientation.scatter(figure=fig_multiple, position=121)

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

    def test_scatter_ipf(self):
        plt.rcParams["axes.grid"] = False

        vx = Vector3d.xvector()
        vz = Vector3d.zvector()

        ori = Orientation.from_euler((325, 48, 163), symmetry=Oh, degrees=True)

        # Returned figure has the expected default properties
        fig_size = (5, 5)
        fig = ori.scatter(
            "ipf", return_figure=True, figure_kwargs=dict(figsize=fig_size)
        )
        assert (fig.get_size_inches() == fig_size).all()
        axes = fig.axes[0]
        assert isinstance(axes, InversePoleFigurePlot)
        assert len(fig.axes) == 1
        assert axes._direction.dot(vz)[0] == 1
        assert axes._hemisphere == "upper"

        # It's possible to add to an existing figure
        ori2 = ori * ori
        ori2.symmetry = ori.symmetry
        fig2 = ori2.scatter("ipf", figure=fig, return_figure=True)
        assert fig == fig2
        assert len(axes.collections) == 2
        # Vectors plotted are inside the fundamental sector
        x, y = axes.collections[0].get_offsets().data.squeeze()
        v1 = axes._projection.inverse.xy2vector(x, y)
        assert v1 < ori.symmetry.fundamental_sector

        # Passing multiple directions yields multiple plots
        fig3 = ori.scatter(
            "ipf", direction=Vector3d.stack((vx, vz)), return_figure=True
        )
        assert len(fig3.axes) == 2

        # Plotting an IPF defined by a sector with vertices on both
        # sides of the equator yields two axes
        ori.symmetry = C4
        fig4 = ori.scatter("ipf", return_figure=True)
        axes4 = fig4.axes
        assert len(axes4) == 2
        # Vector not visible in the lower hemisphere
        assert len(axes4[1].collections) == 0
        x, y = axes4[0].collections[0].get_offsets().data.squeeze()
        v2 = axes4[0]._projection.inverse.xy2vector(x, y)
        assert v2 < ori.symmetry.fundamental_sector

        plt.close("all")

    def test_in_fundamental_region(self):
        # (2 pi, pi, 2 pi) and some random orientations
        ori = Orientation(
            (
                (0, -1, 0, 0),
                (0.4094, 0.7317, -0.4631, -0.2875),
                (-0.3885, 0.5175, -0.7589, 0.0726),
                (-0.5407, -0.7796, 0.2955, -0.1118),
                (-0.3874, 0.6708, -0.1986, 0.6004),
            )
        )
        for pg in _proper_groups:
            ori.symmetry = pg
            region = np.radians(pg.euler_fundamental_region)
            assert np.all(np.max(ori.in_euler_fundamental_region(), axis=0) <= region)


class TestOrientationInitialization:
    def test_from_euler_symmetry(self):
        euler = np.deg2rad([90, 45, 90])
        o1 = Orientation.from_euler(euler)
        assert np.allclose(o1.data, [0, -0.3827, 0, -0.9239], atol=1e-4)
        assert o1.symmetry.name == "1"
        o2 = Orientation.from_euler(euler, symmetry=Oh)
        o2 = o2.reduce()
        assert np.allclose(o2.data, [0.9239, 0, -0.3827, 0], atol=1e-4)
        assert o2.symmetry.name == "m-3m"
        o3 = Orientation(o1.data, symmetry=Oh)
        o3 = o3.reduce()
        assert np.allclose(o3.data, o2.data)

        o4 = Orientation.from_euler(np.rad2deg(euler), degrees=True)
        assert np.allclose(o4.data, o1.data)

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
        o2 = o2.reduce()
        assert np.allclose(
            o2.data, np.array([1, 0, 0, 0] * 2 + [-1, 0, 0, 0] * 2).reshape((4, 4))
        )
        assert o2.symmetry.name == "m-3m"
        o3 = Orientation(o1.data, symmetry=Oh)
        o3 = o3.reduce()
        assert np.allclose(o3.data, o2.data)

    def test_from_align_vectors(self):
        phase = Phase(
            point_group="4",
            structure=Structure(lattice=Lattice(0.5, 0.5, 1, 90, 90, 90)),
        )
        a = Miller(uvw=[[2, -1, 0], [0, 0, 1]], phase=phase)
        b = Vector3d([[3, 1, 0], [-1, 3, 0]])
        ori = Orientation.from_align_vectors(a, b)
        assert type(ori) == Orientation
        assert ori.symmetry == phase.point_group
        assert np.allclose(a.unit.data, (ori * b.unit).data)
        a = Miller([[2, -1, 0], [0, 0, 1]])
        _, e = Orientation.from_align_vectors(a, b, return_rmsd=True)
        assert e == 0
        _, m = Orientation.from_align_vectors(a, b, return_sensitivity=True)
        assert np.allclose(m, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.5]]))
        out = Orientation.from_align_vectors(
            a, b, return_rmsd=True, return_sensitivity=True
        )
        assert len(out) == 3
        a = Vector3d([[2, -1, 0], [0, 0, 1]])
        with pytest.raises(
            ValueError,
            match="Argument other must be of type Miller, but has type "
            "<class 'orix.vector.vector3d.Vector3d'>",
        ):
            _ = Orientation.from_align_vectors(a, b)

    def test_from_neo_euler_symmetry(self):
        v = AxAngle.from_axes_angles(axes=Vector3d.zvector(), angles=np.pi / 2)
        o1 = Orientation.from_neo_euler(v)
        assert np.allclose(o1.data, [0.7071, 0, 0, 0.7071])
        assert o1.symmetry.name == "1"
        o2 = Orientation.from_neo_euler(v, symmetry=Oh)
        o2 = o2.reduce()
        assert np.allclose(o2.data, [-1, 0, 0, 0])
        assert o2.symmetry.name == "m-3m"
        o3 = Orientation(o1.data, symmetry=Oh)
        o3 = o3.reduce()
        assert np.allclose(o3.data, o2.data)

    def test_from_axes_angles(self, rotations):
        axis = Vector3d.xvector() - Vector3d.yvector()
        angle = np.pi / 2
        axangle = AxAngle.from_axes_angles(axis, angle)
        o1 = Orientation.from_neo_euler(axangle, Oh)
        o2 = Orientation.from_axes_angles(axis, angle, Oh)
        assert np.allclose(o1.to_euler(degrees=True), [135, 90, 225])
        assert np.allclose(o1.data, o2.data)
        assert o1.symmetry.name == o2.symmetry.name == "m-3m"
        assert np.allclose(o1.symmetry.data, o2.symmetry.data)

        o3 = Orientation.from_axes_angles(axis, np.rad2deg(angle), Oh, degrees=True)
        assert np.allclose(o2.data, o3.data)

    def test_get_identity(self):
        """Get the identity orientation via two alternative routes."""
        o1 = Orientation([0.4884, 0.1728, 0.2661, 0.8129])
        o2 = Orientation([0.8171, -0.2734, 0.161, -0.4813])

        # Route 1 from a Misorientation instance
        m12_1 = o2 - o1
        o3_1 = (m12_1 * o1) * ~o2

        # Route 2 from a third Orientation instance
        m12_2 = o2 * ~o1
        o3_2 = (m12_2 * o1) * ~o2

        assert np.allclose(m12_1.data, m12_2.data)
        assert np.allclose(o3_1.data, o3_2.data)
        assert np.allclose(o3_1.data, [1, 0, 0, 0])

    def test_from_scipy_rotation(self):
        """Assert correct type and symmetry is returned and that the
        misorientation rotates crystal directions correctly.
        """
        r_scipy = SciPyRotation.from_euler("ZXZ", [90, 0, 0], degrees=True)

        ori1 = Orientation.from_scipy_rotation(r_scipy)
        assert isinstance(ori1, Orientation)
        assert ori1.symmetry.name == "1"

        ori2 = Orientation.from_scipy_rotation(r_scipy, Oh)
        assert np.allclose(ori2.symmetry.data, Oh.data)

        uvw = Miller(uvw=[1, 1, 0], phase=Phase(point_group="m-3m"))
        uvw2 = ori2 * uvw
        assert np.allclose(uvw2.data, [1, -1, 0])

        # Raises an appropriate error message
        with pytest.raises(TypeError, match="Value must be an instance of"):
            _ = Orientation.from_scipy_rotation(r_scipy, (Oh, Oh))

    # TODO: Remove in 1.0
    def test_from_euler_warns(self):
        """Orientation.from_euler() warns only once when "convention"
        argument is passed.
        """
        euler = np.random.rand(10, 3)

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            _ = Orientation.from_euler(euler)

        msg = (
            r"Argument `convention` is deprecated and will be removed in version 1.0. "
            r"To avoid this warning, please do not use `convention`. "
            r"Use `direction` instead. See the documentation of `from_euler\(\)` for "
            "more details."
        )
        with pytest.warns(np.VisibleDeprecationWarning, match=msg) as record2:
            _ = Orientation.from_euler(euler, convention="whatever")
        assert len(record2) == 1

    # TODO: Remove in 1.0
    def test_from_euler_convention_mtex(self):
        """Passing convention="mtex" to Orientation.from_euler() works
        but warns once.
        """
        euler = np.random.rand(10, 3)
        ori1 = Orientation.from_euler(euler, direction="crystal2lab")
        with pytest.warns(np.VisibleDeprecationWarning, match=r"Argument `convention`"):
            ori2 = Orientation.from_euler(euler, convention="mtex")
        assert np.allclose(ori1.data, ori2.data)

    # TODO: Remove in 1.0
    def test_to_euler_convention_warns(self):
        """Orientation.to_euler() warns only once when "convention"
        argument is passed.
        """
        ori1 = Orientation.from_euler(np.random.rand(10, 3))

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            ori2 = ori1.to_euler()

        msg = (
            r"Argument `convention` is deprecated and will be removed in version 1.0. "
            r"To avoid this warning, please do not use `convention`. "
            r"See the documentation of `to_euler\(\)` for more details."
        )
        with pytest.warns(np.VisibleDeprecationWarning, match=msg):
            ori3 = ori1.to_euler(convention="whatever")
        assert np.allclose(ori2, ori3)
