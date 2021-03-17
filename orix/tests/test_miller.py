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

from diffpy.structure import Lattice, Structure
import numpy as np

from orix.crystal_map import Phase
from orix.vector import Miller


class TestMiller:
    pass


# Run tests for all systems: $ pytest -k TestMillerPointGroups
# Run tests for one system:  $ pytest -k TestMillerPointGroupsMonoclinic


class TestMillerPointGroups:
    hkl = [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
    phase = Phase(structure=Structure(lattice=Lattice(1, 1, 1, 90, 90, 90)))


class TestMillerPointGroupsTriclinic(TestMillerPointGroups):
    # Triclinic: 1, -1
    def test_group_1(self):
        self.phase.point_group = "1"
        m = Miller(hkl=self.hkl, phase=self.phase)

        assert np.allclose(m.symmetrise(unique=False).hkl, self.hkl)
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(m_unique.hkl, self.hkl)

        mult = m.multiplicity
        assert np.allclose(mult, [1, 1, 1])
        assert np.sum(mult) == m_unique.size

    def test_group_bar1(self):
        self.phase.point_group = "-1"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1, -1],

                [ 1,  1,  1],
                [-1, -1, -1],
            ],
        )

        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1, -1],

                [ 1,  1,  1],
                [-1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 2, 2])
        assert np.sum(mult) == m_unique.size


class TestMillerPointGroupsMonoclinic(TestMillerPointGroups):
    # Monoclinic: 2 (121), m (1m1), 2/m
    def test_group_121(self):
        self.phase.point_group = "121"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1,  1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1,  1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 2, 2])
        assert np.sum(mult) == m_unique.size

    def test_group_1m1(self):
        self.phase.point_group = "1m1"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],

                [ 0,  1,  1],
                [ 0, -1,  1],

                [ 1,  1,  1],
                [ 1, -1,  1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],

                [ 0,  1,  1],
                [ 0, -1,  1],

                [ 1,  1,  1],
                [ 1, -1,  1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [1, 2, 2])
        assert np.sum(mult) == m_unique.size

    def test_group_2overm(self):
        self.phase.point_group = "2/m"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0,  1, -1],
                [ 0, -1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1,  1, -1],
                [-1, -1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0,  1, -1],
                [ 0, -1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1,  1, -1],
                [-1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 4, 4])
        assert np.sum(mult) == m_unique.size


class TestMillerPointGroupsOrthorhombic(TestMillerPointGroups):
    # Orthorhombic: 222, mm2, 2/m 2/m 2/m (mmm)
    def test_group_222(self):
        self.phase.point_group = "222"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 4, 4])
        assert np.sum(mult) == m_unique.size

    def test_group_mm2(self):
        self.phase.point_group = "mm2"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0,  1],

                [ 0,  1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],
                [ 0, -1,  1],

                [ 1,  1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1, -1,  1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],
                [ 0, -1,  1],

                [ 1,  1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1, -1,  1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 4, 4])
        assert np.sum(mult) == m_unique.size

    def test_group_2overm_2overm_2overm(self):
        self.phase.point_group = "mmm"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0,  1, -1],
                [ 0,  1,  1],
                [ 0,  1, -1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 0, -1,  1],
                [ 0, -1, -1],

                [ 1,  1,  1],
                [ 1,  1, -1],
                [-1,  1,  1],
                [-1,  1, -1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [-1, -1,  1],
                [-1, -1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0,  1, -1],
                [ 0, -1,  1],
                [ 0, -1, -1],

                [ 1,  1,  1],
                [ 1,  1, -1],
                [-1,  1,  1],
                [-1,  1, -1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [-1, -1,  1],
                [-1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 4, 8])
        assert np.sum(mult) == m_unique.size


class TestMillerPointGroupsTetragonal(TestMillerPointGroups):
    # Tetragonal: 4, -4, 4/m, 422, 4mm, -42m, 4/m 2/m 2/m
    def test_group_4(self):
        self.phase.point_group = "4"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [1, 4, 4])
        assert np.sum(mult) == m_unique.size

    def test_group_bar4(self):
        self.phase.point_group = "-4"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [-1, -1, -1],
                [ 1,  1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [-1, -1, -1],
                [ 1,  1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 4, 4])
        assert np.sum(mult) == m_unique.size

    def test_group_4overm(self):
        self.phase.point_group = "4/m"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 8, 8])
        assert np.sum(mult) == m_unique.size

    def test_group_422(self):
        self.phase.point_group = "422"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 8, 8])
        assert np.sum(mult) == m_unique.size

    def test_group_4mm(self):
        self.phase.point_group = "4mm"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [1, 4, 4])
        assert np.sum(mult) == m_unique.size

    def test_group_bar42m(self):
        self.phase.point_group = "-42m"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],
                [ 1,  0,  1],
                [-1,  0,  1],
                [ 1,  0, -1],
                [-1,  0, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 0,  1, -1],
                [ 1,  0,  1],
                [-1,  0,  1],
                [ 1,  0, -1],
                [-1,  0, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 8, 4])
        assert np.sum(mult) == m_unique.size

    def test_group_4overm_2overm_2overm(self):
        self.phase.point_group = "4/mmm"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 8, 8])
        assert np.sum(mult) == m_unique.size


class TestMillerPointGroupsTrigonal(TestMillerPointGroups):
    # Trigonal: 3, -3, 32, 3m, -3 2/m
    def test_group_3(self):
        self.phase.point_group = "3"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [1, 3, 3])
        assert np.sum(mult) == m_unique.size

    def test_group_bar3(self):
        self.phase.point_group = "-3"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 6, 6])
        assert np.sum(mult) == m_unique.size

    def test_group_32(self):
        self.phase.point_group = "32"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 6, 6])
        assert np.sum(mult) == m_unique.size

    def test_group_3m(self):
        self.phase.point_group = "3m"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],
                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
                [-1    ,  1    , 1],
                [-0.366, -1.366, 1],
                [ 1.366,  0.366, 1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
                [-1    ,  1    , 1],
                [-0.366, -1.366, 1],
                [ 1.366,  0.366, 1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [1, 3, 6])
        assert np.sum(mult) == m_unique.size

    def test_group_bar3_2overm(self):
        self.phase.point_group = "-3m"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],
                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
                [-1    ,  1    ,  1],
                [-0.366, -1.366,  1],
                [ 1.366,  0.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
                [-1    ,  1    ,  1],
                [-0.366, -1.366,  1],
                [ 1.366,  0.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 6, 12])
        assert np.sum(mult) == m_unique.size


class TestMillerPointGroupsHexagonal(TestMillerPointGroups):
    # Hexagonal: 6, -6, 6/m, 622, 6mm, -6m2, 6/m 2/m 2/m (6/mmm)
    def test_group_6(self):
        self.phase.point_group = "6"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],
                [ 0    , -1    , 1],
                [ 0.866,  0.5  , 1],
                [-0.866,  0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
                [-1    , -1    , 1],
                [ 1.366, -0.366, 1],
                [-0.366,  1.366, 1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],
                [ 0    , -1    , 1],
                [ 0.866,  0.5  , 1],
                [-0.866,  0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
                [-1    , -1    , 1],
                [ 1.366, -0.366, 1],
                [-0.366,  1.366, 1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [1, 6, 6])
        assert np.sum(mult) == m_unique.size

    def test_group_bar6(self):
        self.phase.point_group = "-6"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 6, 6])
        assert np.sum(mult) == m_unique.size

    def test_group_6overm(self):
        self.phase.point_group = "6/m"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    ,  1],
                [ 0.866,  0.5  ,  1],
                [-0.866,  0.5  ,  1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    ,  1],
                [ 1.366, -0.366,  1],
                [-0.366,  1.366,  1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    ,  1],
                [ 0.866,  0.5  ,  1],
                [-0.866,  0.5  ,  1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    ,  1],
                [ 1.366, -0.366,  1],
                [-0.366,  1.366,  1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 12, 12])
        assert np.sum(mult) == m_unique.size

    def test_group_622(self):
        self.phase.point_group = "622"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    ,  1],
                [ 0.866,  0.5  ,  1],
                [-0.866,  0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    ,  1],
                [ 1.366, -0.366,  1],
                [-0.366,  1.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
                [-1    ,  1    , -1],
                [-0.366, -1.366, -1],
                [ 1.366,  0.366, -1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    ,  1],
                [ 0.866,  0.5  ,  1],
                [-0.866,  0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    ,  1],
                [ 1.366, -0.366,  1],
                [-0.366,  1.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
                [-1    ,  1    , -1],
                [-0.366, -1.366, -1],
                [ 1.366,  0.366, -1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 12, 12])
        assert np.sum(mult) == m_unique.size

    def test_group_6mm(self):
        self.phase.point_group = "6mm"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],
                [ 0    , -1    , 1],
                [ 0.866,  0.5  , 1],
                [-0.866,  0.5  , 1],
                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],
                [ 0    , -1    , 1],
                [ 0.866,  0.5  , 1],
                [-0.866,  0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
                [-1    , -1    , 1],
                [ 1.366, -0.366, 1],
                [-0.366,  1.366, 1],
                [-1    ,  1    , 1],
                [-0.366, -1.366, 1],
                [ 1.366,  0.366, 1],
                [ 1    , -1    , 1],
                [ 0.366,  1.366, 1],
                [-1.366, -0.366, 1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    , 1],

                [ 0    ,  1    , 1],
                [-0.866, -0.5  , 1],
                [ 0.866, -0.5  , 1],
                [ 0    , -1    , 1],
                [ 0.866,  0.5  , 1],
                [-0.866,  0.5  , 1],

                [ 1    ,  1    , 1],
                [-1.366,  0.366, 1],
                [ 0.366, -1.366, 1],
                [-1    , -1    , 1],
                [ 1.366, -0.366, 1],
                [-0.366,  1.366, 1],
                [-1    ,  1    , 1],
                [-0.366, -1.366, 1],
                [ 1.366,  0.366, 1],
                [ 1    , -1    , 1],
                [ 0.366,  1.366, 1],
                [-1.366, -0.366, 1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [1, 6, 12])
        assert np.sum(mult) == m_unique.size

    def test_group_bar6m2(self):
        self.phase.point_group = "-6m2"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],
                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    ,  1    , -1],
                [-0.366, -1.366, -1],
                [ 1.366,  0.366, -1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
                [-1    ,  1    ,  1],
                [-0.366, -1.366,  1],
                [ 1.366,  0.366,  1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    ,  1    , -1],
                [-0.366, -1.366, -1],
                [ 1.366,  0.366, -1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
                [-1    ,  1    ,  1],
                [-0.366, -1.366,  1],
                [ 1.366,  0.366,  1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 6, 12])
        assert np.sum(mult) == m_unique.size

    def test_group_6overm_2overm_2overm(self):
        self.phase.point_group = "6/mmm"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    , -1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],
                [ 0    ,  0    ,  1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    ,  1],
                [ 0.866,  0.5  ,  1],
                [-0.866,  0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],
                [ 0    , -1    ,  1],
                [ 0.866,  0.5  ,  1],
                [-0.866,  0.5  ,  1],
                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    ,  1],
                [ 1.366, -0.366,  1],
                [-0.366,  1.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
                [-1    ,  1    , -1],
                [-0.366, -1.366, -1],
                [ 1.366,  0.366, -1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
                [ 1    , -1    ,  1],
                [ 0.366,  1.366,  1],
                [-1.366, -0.366,  1],
                [-1    ,  1    ,  1],
                [-0.366, -1.366,  1],
                [ 1.366,  0.366,  1],
            ],
            atol=1e-3
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0    ,  0    ,  1],
                [ 0    ,  0    , -1],

                [ 0    ,  1    ,  1],
                [-0.866, -0.5  ,  1],
                [ 0.866, -0.5  ,  1],
                [ 0    , -1    ,  1],
                [ 0.866,  0.5  ,  1],
                [-0.866,  0.5  ,  1],
                [ 0    , -1    , -1],
                [ 0.866,  0.5  , -1],
                [-0.866,  0.5  , -1],
                [ 0    ,  1    , -1],
                [-0.866, -0.5  , -1],
                [ 0.866, -0.5  , -1],

                [ 1    ,  1    ,  1],
                [-1.366,  0.366,  1],
                [ 0.366, -1.366,  1],
                [-1    , -1    ,  1],
                [ 1.366, -0.366,  1],
                [-0.366,  1.366,  1],
                [ 1    , -1    , -1],
                [ 0.366,  1.366, -1],
                [-1.366, -0.366, -1],
                [-1    ,  1    , -1],
                [-0.366, -1.366, -1],
                [ 1.366,  0.366, -1],
                [ 1    ,  1    , -1],
                [-1.366,  0.366, -1],
                [ 0.366, -1.366, -1],
                [-1    , -1    , -1],
                [ 1.366, -0.366, -1],
                [-0.366,  1.366, -1],
                [ 1    , -1    ,  1],
                [ 0.366,  1.366,  1],
                [-1.366, -0.366,  1],
                [-1    ,  1    ,  1],
                [-0.366, -1.366,  1],
                [ 1.366,  0.366,  1],
            ],
            atol=1e-3
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [2, 12, 24])
        assert np.sum(mult) == m_unique.size


class TestMillerPointGroupsCubic(TestMillerPointGroups):
    # Cubic: 23, 2/m -3, 432, -43m, 4/m -3 2/m (m-3m)
    def test_group_23(self):
        self.phase.point_group = "23"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 0,  1,  0],
                [ 0, -1,  0],
                [ 0, -1,  0],
                [ 0,  1,  0],
                [ 0,  0, -1],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [-1,  0,  1],
                [ 1,  0, -1],
                [-1,  0, -1],
                [ 1,  1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],
                [-1,  1,  0],
                [ 0, -1, -1],
                [ 0,  1, -1],
                [ 1,  1,  1],

                [-1, -1,  1],
                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [ 1, -1, -1],
                [-1,  1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 0,  1,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [-1,  0,  1],
                [ 1,  0, -1],
                [-1,  0, -1],
                [ 1,  1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],
                [-1,  1,  0],
                [ 0, -1, -1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [6, 12, 4])
        assert np.sum(mult) == m_unique.size

    def test_group_2overm_bar3(self):
        self.phase.point_group = "m-3"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 0,  1,  0],
                [ 0, -1,  0],
                [ 0, -1,  0],
                [ 0,  1,  0],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 1,  0,  0],
                [ 0, -1,  0],
                [ 0,  1,  0],
                [ 0,  1,  0],
                [ 0, -1,  0],
                [ 0,  0,  1],
                [ 0,  0,  1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [-1,  0,  1],
                [ 1,  0, -1],
                [-1,  0, -1],
                [ 1,  1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],
                [-1,  1,  0],
                [ 0, -1, -1],
                [ 0,  1, -1],
                [ 0, -1, -1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 1,  0, -1],
                [-1,  0,  1],
                [ 1,  0,  1],
                [-1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
                [ 1, -1,  0],
                [ 0,  1,  1],
                [ 0, -1,  1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1,  1, -1],
                [-1, -1, -1],
                [ 1,  1, -1],
                [-1,  1,  1],
                [ 1, -1,  1],
                [-1, -1, -1],
                [ 1,  1, -1],
                [-1,  1,  1],
                [ 1, -1,  1],
                [-1,  1,  1],
                [ 1, -1,  1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 0,  1,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [-1,  0,  1],
                [ 1,  0, -1],
                [-1,  0, -1],
                [ 1,  1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],
                [-1,  1,  0],
                [ 0, -1, -1],
                [ 0,  1, -1],

                [ 1,  1,  1],
                [-1, -1,  1],
                [ 1, -1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1,  1, -1],
                [-1,  1,  1],
                [ 1, -1,  1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [6, 12, 8])
        assert np.sum(mult) == m_unique.size

    def test_group_432(self):
        self.phase.point_group = "432"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 1,  0,  1],
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
                [-1, -1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
                [-1, -1,  0],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [6, 12, 8])
        assert np.sum(mult) == m_unique.size

    def test_group_4overm_bar3_2overm(self):
        self.phase.point_group = "m-3m"
        m = Miller(hkl=self.hkl, phase=self.phase)

        # fmt: off
        assert np.allclose(
            m.symmetrise(unique=False).hkl,
            [
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [ 0,  0, -1],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 0,  0,  1],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 1,  0,  1],
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
                [-1, -1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [-1,  0, -1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 1,  0,  1],
                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [-1,  1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
                [-1, -1,  0],
                [ 1, -1,  0],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [ 1, -1,  1],
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [-1,  1, -1],
                [-1, -1, -1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
            ],
        )
        m_unique = m.symmetrise(unique=True)
        assert np.allclose(
            m_unique.hkl,
            [
                [ 0,  0,  1],
                [ 1,  0,  0],
                [ 0,  1,  0],
                [-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1],

                [ 0,  1,  1],
                [-1,  0,  1],
                [ 0, -1,  1],
                [ 1,  0,  1],
                [ 0, -1, -1],
                [ 1,  0, -1],
                [ 0,  1, -1],
                [-1,  0, -1],
                [ 1, -1,  0],
                [ 1,  1,  0],
                [-1,  1,  0],
                [-1, -1,  0],

                [ 1,  1,  1],
                [-1,  1,  1],
                [-1, -1,  1],
                [ 1, -1,  1],
                [ 1, -1, -1],
                [ 1,  1, -1],
                [-1,  1, -1],
                [-1, -1, -1],
            ],
        )
        # fmt: on

        mult = m.multiplicity
        assert np.allclose(mult, [6, 12, 8])
        assert np.sum(mult) == m_unique.size
