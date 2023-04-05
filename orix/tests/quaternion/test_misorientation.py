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

from diffpy.structure import Lattice, Structure
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as SciPyRotation

# fmt: off
# isort: off
from orix.crystal_map import Phase
from orix.quaternion import Misorientation
from orix.quaternion.symmetry import (
    C1,
    C2,
    C4,
    C3,
    D3,
    D6,
    O,
    Oh,
    _groups,
)
from orix.vector import Miller, Vector3d
# isort: on
# fmt: on


class TestMisorientation:
    @pytest.mark.parametrize("symmetry", ([C2, C3], [Oh, C2], [O, D3]))
    def test_reshape_maintains_symmetry(self, symmetry):
        m = Misorientation.random((4, 5))
        m.symmetry = symmetry
        m1 = m.reshape(5, 4)
        for s1, s2 in zip(m1.symmetry, symmetry):
            assert s1._tuples == s2._tuples

    @pytest.mark.parametrize("symmetry", ([C2, C3], [Oh, C2], [O, D3]))
    def test_transpose_maintains_symmetry(self, symmetry):
        m = Misorientation.random((4, 5))
        m.symmetry = symmetry
        m1 = m.transpose()
        for s1, s2 in zip(m1.symmetry, symmetry):
            assert s1._tuples == s2._tuples

    @pytest.mark.parametrize("symmetry", ([C2, C3], [Oh, C2], [O, D3]))
    def test_flatten_maintains_symmetry(self, symmetry):
        m = Misorientation.random((4, 5))
        m.symmetry = symmetry
        m1 = m.flatten()
        for s1, s2 in zip(m1.symmetry, symmetry):
            assert s1._tuples == s2._tuples

    @pytest.mark.parametrize("symmetry", ([C2, C3], [Oh, C2], [O, D3]))
    def test_squeeze_maintains_symmetry(self, symmetry):
        m = Misorientation.random((4, 5, 1))
        m.symmetry = symmetry
        m1 = m.squeeze()
        for s1, s2 in zip(m1.symmetry, symmetry):
            assert s1._tuples == s2._tuples

    @pytest.mark.parametrize("Gl", [C4, C2])
    def test_equivalent(self, Gl):
        """Tests that the property Misorientation.equivalent runs without error,
        use grain_exchange=True as this falls back to grain_exchange=False when
        Gl!=Gr:

        Gl == C4 is grain exchange
        Gl == C2 is no grain exchange
        """
        m = Misorientation([1, 1, 1, 1])  # any will do
        m_new = Misorientation(m.data, symmetry=(Gl, C4))
        m_new = m_new.map_into_symmetry_reduced_zone()
        _ = m_new.equivalent(grain_exchange=True)

    def test_string_representation(self):
        m = Misorientation([1, 1, 1, 1])  # any will do
        _ = repr(m)

    def test_symmetry_property(self):
        m = Misorientation.random((3, 2))
        m.symmetry = (Oh, C3)
        assert m.symmetry == (Oh, C3)
        assert m._symmetry == (Oh, C3)

    @pytest.mark.parametrize(
        "error_type, value",
        [(ValueError, (1, 2)), (ValueError, (C1, 2)), (TypeError, 1)],
    )
    def test_symmetry_property_wrong_type(self, error_type, value):
        mori = Misorientation.random((3, 2))
        with pytest.raises(error_type, match="Value must be a 2-tuple"):
            mori.symmetry = value

    @pytest.mark.parametrize(
        "error_type, value",
        [(ValueError, (C1,)), (ValueError, (C1, C2, C1))],
    )
    def test_symmetry_property_wrong_number_of_values(self, error_type, value):
        o = Misorientation.random((3, 2))
        with pytest.raises(error_type, match="Value must be a 2-tuple"):
            # less than 2 Symmetry
            o.symmetry = value

    def test_get_distance_matrix(self):
        """Compute distance between every misorientation in an instance
        with every other misorientation in the same instance.

        Misorientations are taken from the misorientation clustering
        user guide.
        """
        m1 = Misorientation(
            [
                [-0.8541, -0.5201, -0.0053, -0.0002],
                [-0.8486, -0.5291, -0.0019, -0.0018],
                [-0.7851, -0.6194, -0.0043, -0.0004],
                [-0.7802, -0.3136, -0.5413, -0.0029],
                [-0.8518, -0.5237, -0.0004, -0.0102],
            ],
            symmetry=(D6, D6),
        )
        distance1 = m1.get_distance_matrix()
        assert np.allclose(np.diag(distance1), 0)
        expected = np.array(
            [
                [0, 0.0224, 0.2420, 0.2580, 0.0239],
                [0.0224, 0, 0.2210, 0.2367, 0.0212],
                [0.2419, 0.2209, 0, 0.0184, 0.2343],
                [0.2579, 0.2367, 0.0184, 0, 0.2496],
                [0.0239, 0.0212, 0.2343, 0.2497, 0],
            ]
        )
        assert np.allclose(distance1, expected, atol=1e-4)

        distance2 = m1.get_distance_matrix(degrees=True)
        assert np.allclose(np.rad2deg(distance1), distance2)

    def test_get_distance_matrix_shape(self):
        shape = (3, 4)
        m2 = Misorientation.random(shape)
        distance2 = m2.get_distance_matrix()
        assert distance2.shape == 2 * shape

    def test_get_distance_matrix_progressbar_chunksize(self):
        m = Misorientation.random((5, 15, 4))
        angle1 = m.get_distance_matrix(chunk_size=5)
        angle2 = m.get_distance_matrix(chunk_size=10, progressbar=False)
        assert np.allclose(angle1, angle2)

    @pytest.mark.parametrize("symmetry", _groups[:-1])
    def test_get_distance_matrix_equal_explicit_calculation(self, symmetry):
        # do not test Oh, as this takes ~4 GB
        m = Misorientation.random((5,))
        m.symmetry = (symmetry, symmetry)
        angle1 = m.get_distance_matrix()
        s1, s2 = m.symmetry
        # computation of mismisorientation
        # eq 6 in Johnstone et al. 2020
        p1 = s1.outer(m).outer(s2)
        p2 = s1.outer(~m).outer(s2)
        # for identical symmetries this is equivalent to the old
        # distance function:
        # d = s2.outer(~m).outer(s1.outer(s1)).outer(m).outer(s2)
        p12 = p1.outer(p2)
        angle2 = p12.angle.min(axis=(0, 2, 3, 5))
        assert np.allclose(angle1, angle2)

    def test_from_align_vectors(self):
        phase = Phase(
            point_group="4",
            structure=Structure(lattice=Lattice(0.5, 0.5, 1, 90, 90, 90)),
        )
        a = Miller(uvw=[[2, -1, 0], [0, 0, 1]], phase=phase)
        b = Miller(uvw=[[3, 1, 0], [-1, 3, 0]], phase=phase)
        ori = Misorientation.from_align_vectors(a, b)
        assert type(ori) == Misorientation
        assert ori.symmetry == (phase.point_group,) * 2
        assert np.allclose(a.unit.data, (ori * b.unit).data)
        a = Miller([[2, -1, 0], [0, 0, 1]])
        b = Miller([[3, 1, 0], [-1, 3, 0]])
        _, e = Misorientation.from_align_vectors(a, b, return_rmsd=True)
        assert e == 0
        _, m = Misorientation.from_align_vectors(a, b, return_sensitivity=True)
        assert np.allclose(m, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.5]]))
        out = Misorientation.from_align_vectors(
            a, b, return_rmsd=True, return_sensitivity=True
        )
        assert len(out) == 3
        a = Vector3d([[2, -1, 0], [0, 0, 1]])
        with pytest.raises(
            ValueError,
            match="Arguments other and initial must both be of type Miller, "
            "but are of type <class 'orix.vector.vector3d.Vector3d'> and "
            "<class 'orix.vector.miller.Miller'>.",
        ):
            _ = Misorientation.from_align_vectors(a, b)

    def test_from_scipy_rotation(self):
        """Assert correct type and symmetry is returned and that the
        misorientation rotates crystal directions correctly.
        """
        r_scipy = SciPyRotation.from_euler("ZXZ", [90, 0, 0], degrees=True)

        mori1 = Misorientation.from_scipy_rotation(r_scipy)
        assert isinstance(mori1, Misorientation)
        assert mori1.symmetry[0].name == "1"
        assert mori1.symmetry[1].name == "1"

        mori2 = Misorientation.from_scipy_rotation(r_scipy, (Oh, Oh))
        assert np.allclose(mori2.symmetry[0].data, Oh.data)
        assert np.allclose(mori2.symmetry[1].data, Oh.data)

        uvw = Miller(uvw=[1, 1, 0], phase=Phase(point_group="m-3m"))
        uvw2 = mori2 * uvw
        assert np.allclose(uvw2.data, [1, -1, 0])
        uvw3 = ~mori2 * uvw
        assert np.allclose(uvw3.data, [-1, 1, 0])

        # Raises
        with pytest.raises(TypeError, match="Value must be a 2-tuple of"):
            _ = Misorientation.from_scipy_rotation(r_scipy, Oh)
