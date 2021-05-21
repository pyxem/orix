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

import numpy as np
import pytest

from orix.quaternion.orientation import Orientation, Misorientation
from orix.quaternion.symmetry import C1, C2, C3, C4, D2, D3, D6, T, O, Oh
from orix.vector import AxAngle, Vector3d


@pytest.fixture
def vector(request):
    return Vector3d(request.param)


@pytest.fixture(params=[(0.5, 0.5, 0.5, 0.5), (0.5 ** 0.5, 0, 0, 0.5 ** 0.5)])
def orientation(request):
    return Orientation(request.param)


class TestOrientation:
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

    @pytest.mark.parametrize(
        "orientation, symmetry, expected",
        [
            # fmt: off
            ((1, 0, 0, 0), C1, (1, 0, 0, 0)),
            ((1, 0, 0, 0), C4, (1, 0, 0, 0)),
            ((1, 0, 0, 0), D3, (1, 0, 0, 0)),
            ((1, 0, 0, 0), T,  (1, 0, 0, 0)),
            ((1, 0, 0, 0), O,  (1, 0, 0, 0)),
            # 7pi/12 -C2-> # 7pi/12
            ((0.6088, 0, 0, 0.7934), C2, (-0.7934, 0, 0,  0.6088)),
            # 7pi/12 -C3-> # 7pi/12
            ((0.6088, 0, 0, 0.7934), C3, (-0.9914, 0, 0,  0.1305)),
            # 7pi/12 -C4-> # pi/12
            ((0.6088, 0, 0, 0.7934), C4, (-0.9914, 0, 0, -0.1305)),
            # 7pi/12 -O-> # pi/12
            ((0.6088, 0, 0, 0.7934), O , (-0.9914, 0, 0, -0.1305)),
            # fmt: on
        ],
        indirect=["orientation"],
    )
    def test_set_symmetry(self, orientation, symmetry, expected):
        o = orientation.set_symmetry(symmetry)
        assert np.allclose(o.data, expected, atol=1e-3)

    @pytest.mark.parametrize(
        "symmetry, vector",
        [(C1, (1, 2, 3)), (C2, (1, -1, 3)), (C3, (1, 1, 1)), (O, (0, 1, 0))],
        indirect=["vector"],
    )
    def test_orientation_persistence(self, symmetry, vector):
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
            (
                [(1, 0, 0, 0), (0.7071, 0.7071, 0, 0)],
                C1,
                [[0, np.pi / 2], [np.pi / 2, 0]],
            ),
            (
                [(1, 0, 0, 0), (0.7071, 0.7071, 0, 0)],
                C4,
                [[0, np.pi / 2], [np.pi / 2, 0]],
            ),
            ([(1, 0, 0, 0), (0.7071, 0, 0, 0.7071)], C4, [[0, 0], [0, 0]]),
            (
                [
                    [(1, 0, 0, 0), (0.7071, 0, 0, 0.7071)],
                    [(0, 0, 0, 1), (0.9239, 0, 0, 0.3827)],
                ],
                C4,
                [
                    [[[0, 0], [0, np.pi / 4]], [[0, 0], [0, np.pi / 4]]],
                    [
                        [[0, 0], [0, np.pi / 4]],
                        [[np.pi / 4, np.pi / 4], [np.pi / 4, 0]],
                    ],
                ],
            ),
        ],
        indirect=["orientation"],
    )
    def test_distance(self, orientation, symmetry, expected):
        o = orientation.set_symmetry(symmetry)
        distance = o.distance(verbose=True)
        assert np.allclose(distance, expected, atol=1e-3)

    @pytest.mark.parametrize("symmetry", [C1, C2, C4, D2, D6, T, O])
    def test_getitem(self, orientation, symmetry):
        o = orientation.set_symmetry(symmetry)
        assert o[0].symmetry._tuples == symmetry._tuples

    def test_symmetry_is_preserved(self):
        o = Orientation.identity().set_symmetry(O)
        o_inv = ~o
        assert o_inv.symmetry.name == O.name
        assert np.allclose(o.symmetry.data, o_inv.symmetry.data)

    def test_repr(self):
        shape = (2, 3)
        o = Orientation.identity(shape).set_symmetry(O)
        assert repr(o).split("\n")[0] == f"Orientation {shape} {O.name}"


class TestMisorientation:
    @pytest.mark.parametrize("Gl", [C4, C2])
    def test_equivalent(self, Gl):
        """Tests that the property Misorientation.equivalent runs
        without error, use grain_exchange=True as this falls back to
        grain_exchange=False when Gl!=Gr:
            - Gl == C4 is grain exchange
            - Gl == C2 is no grain exchange
        """
        m = Misorientation([1, 1, 1, 1])  # any will do
        m_new = m.set_symmetry(Gl, C4, verbose=True)
        assert len(m_new.symmetry) == 2
        _ = m_new.equivalent(grain_exchange=True)

    def test_sub(self):
        o = Orientation([1, 1, 1, 1])  # any will do
        o = o.set_symmetry(C4)  # only one as it a O
        m = o - o
        assert np.allclose(m.data, [1, 0, 0, 0])

        # Symmetries are preserved
        sym = m.symmetry
        assert sym[0].name == C4.name
        assert sym[1].name == C4.name
        assert np.allclose(sym[0].data, C4.data)

    def test_sub_orientation_and_other(self):
        m = Orientation([1, 1, 1, 1])  # any will do
        with pytest.raises(TypeError):
            _ = m - 3

    def test_repr(self):
        shape = (2, 3)
        m = Misorientation.identity(shape).set_symmetry(C1, O)
        assert repr(m).split("\n")[0] == f"Misorientation {shape} {C1.name}, {O.name}"
