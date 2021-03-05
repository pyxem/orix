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

from orix.crystal_map import Phase
from orix.vector import Miller


class TestMiller:
    # Tested against the EMstar command line program in EMsoft 5.0
    # (https://github.com/EMsoft-org/EMsoft/wiki/EMstar)
    def test_triclinic(self):
        m = Miller([[0, 0, 1], [0, 1, 1], [1, 1, 1]], Phase(space_group=2))
        assert np.allclose(m.multiplicity, [2, 2, 2])
        # [0, 0, 1]
        assert np.allclose(m[0].symmetrise().hkl, [[0, 0, 1], [0, 0, -1]])
        # [0, 1, 1]
        assert np.allclose(m[1].symmetrise().hkl, [[0, 1, 1], [0, -1, -1]])
        # [1, 1, 1]
        assert np.allclose(m[2].symmetrise().hkl, [[1, 1, 1], [-1, -1, -1]])

    def test_monoclinic(self):
        m = Miller([[0, 0, 1], [0, 1, 1], [1, 1, 1]], Phase(space_group=15))
        assert np.allclose(m.multiplicity, [2, 4, 4])
        # [0, 0, 1]
        assert np.allclose(m[0].symmetrise().hkl, [[0, 0, 1], [0, 0, -1]])
        # fmt: off
        # [0, 1, 1]
        assert np.allclose(
            m[1].symmetrise().hkl,
            [
                [0,  1,  1],
                [0,  1, -1],
                [0, -1, -1],
                [0, -1,  1]
            ]
        )
        # [1, 1, 1]
        assert np.allclose(
            m[2].symmetrise().hkl,
            [
                [ 1,  1,  1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1,  1],
            ]
        )
        # fmt: on

    def test_orthorhombic(self):
        m = Miller([[0, 0, 1], [0, 1, 1], [1, 1, 1]], Phase(space_group=61))
        assert np.allclose(m.multiplicity, [2, 4, 8])
        # [0, 0, 1]
        assert np.allclose(m[0].symmetrise().hkl, [[0, 0, 1], [0, 0, -1]])
        # fmt: off
        # [0, 1, 1]
        assert np.allclose(
            m[1].symmetrise().hkl,
            [
                [0,  1,  1],
                [0, -1,  1],
                [0, -1,  1],
                [0, -1, -1],
            ]
        )
        # [1, 1, 1]
        assert np.allclose(
            m[2].symmetrise().hkl,
            [
                [ 1,  1,  1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1, -1,  1],
                [-1,  1,  1],
            ]
        )
        # fmt: on

    def test_tetragonal(self):
        m = Miller([[0, 0, 1], [0, 1, 1], [1, 1, 1]], Phase(space_group=136))
        assert np.allclose(m.multiplicity, [2, 8, 8])
        # [0, 0, 1]
        assert np.allclose(m[0].symmetrise().hkl, [[0, 0, 1], [0, 0, -1]])
        # fmt: off
        # [0, 1, 1]
        assert np.allclose(
            m[1].symmetrise().hkl,
            [
                [ 0,  1,  1],
                [ 0, -1,  1],
                [-1,  0,  1],
                [ 0,  1, -1],
                [ 0, -1, -1],
                [ 1,  0,  1],
                [ 1,  0, -1],
                [-1,  0, -1],
            ]
        )
        # [1, 1, 1]
        assert np.allclose(
            m[2].symmetrise().hkl,
            [
                [ 1,  1,  1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1, -1,  1],
                [-1,  1,  1],
            ]
        )
        # fmt: on

    def test_trigonal(self):
        m = Miller([[0, 0, 1], [0, 1, 1], [1, 1, 1]], Phase(space_group=167))
        assert np.allclose(m.multiplicity, [2, 6, 12])
        # [0, 0, 1]
        assert np.allclose(m[0].symmetrise().hkl, [[0, 0, 1], [0, 0, -1]])
        # fmt: off
        # [0, 1, 1]
        assert np.allclose(
            m[1].symmetrise().hkl,
            [
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 1,  0, -1],
                [ 0, -1, -1],
                [ 1, -1,  1],
                [-1,  1, -1],
            ]
        )
        # [1, 1, 1]
        assert np.allclose(
            m[2].symmetrise().hkl,
            [
                [ 1,  1,  1],
                [-2,  1,  1],
                [ 1, -1,  1],
                [-1, -1, -1],
                [ 1, -2,  1],
                [ 1, -2, -1],
                [ 2, -1, -1],
                [-2,  1, -1],
                [-1,  2, -1],
                [-1, -1,  1],
                [ 2, -1,  1],
                [-1,  2,  1],
            ]
        )
        # fmt: on

    def test_hexagonal(self):
        m = Miller([[0, 0, 1], [0, 1, 1], [1, 1, 1]], Phase(space_group=186))
        assert np.allclose(m.multiplicity, [1, 6, 6])
        # [0, 0, 1]
        assert np.allclose(m[0].symmetrise().hkl, [0, 0, 1])
        # fmt: off
        # [0, 1, 1]
        assert np.allclose(
            m[1].symmetrise().hkl,
            [
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1, -1,  1],
                [ 1,  0,  1],
                [-1,  1,  1],
            ]
        )
        # [1, 1, 1]
        assert np.allclose(
            m[2].symmetrise().hkl,
            [
                [ 1,  1,  1],
                [-2,  1,  1],
                [-1, -1,  1],
                [ 1, -2,  1],
                [ 2, -1,  1],
                [-1,  2,  1],
            ]
        )
        # fmt: on

    def test_cubic(self):
        m = Miller([[0, 0, 1], [0, 1, 1], [1, 1, 1]], Phase(space_group=225))
        assert np.allclose(m.multiplicity, [6, 12, 8])
        # fmt: off
        # [0, 0, 1]
        assert np.allclose(
            m[0].symmetrise().hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0,  1,  0],
            ]
        )
        # [0, 1, 1]
        assert np.allclose(
            m[1].symmetrise().hkl,
            [
                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0,  1, -1],
                [ 1,  0,  1],
                [ 1,  0, -1],
                [ 0, -1, -1],
                [ 1,  1,  0],
                [-1,  0, -1],
                [-1,  1,  0],
                [-1,  0,  1],
                [ 1, -1,  0],
                [-1, -1,  0],
            ]
        )
        # [1, 1, 1]
        assert np.allclose(
            m[2].symmetrise().hkl,
            [
                [ 1,  1,  1],
                [-1, -1,  1],
                [-1,  1, -1],
                [ 1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1, -1,  1],
                [-1,  1,  1],
            ]
        )
        # fmt: on
