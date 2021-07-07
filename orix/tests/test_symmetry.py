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

from diffpy.structure.spacegroups import GetSpaceGroup
import numpy as np
import pytest

# fmt: off
from orix.quaternion.symmetry import (
    C1, Ci,  # triclinic
    C2x, C2y, C2z, Csx, Csy, Csz, Cs, C2, C2h,  # monoclinic
    D2, C2v, D2h,  # orthorhombic
    C4, S4, C4h, D4, C4v, D2d, D4h,  # tetragonal
    C3, S6, D3y, D3, C3v, D3d,  # trigonal
    C6, C3h, C6h, D6, C6v, D3h, D6h,  # hexagonal
    T, Th, O, Td, Oh,  # cubic
    spacegroup2pointgroup_dict,
)
# fmt: on
from orix.quaternion import get_point_group, Rotation, Symmetry
from orix.vector import Vector3d


@pytest.fixture(params=[(1, 2, 3)])
def vector(request):
    return Vector3d(request.param)


@pytest.mark.parametrize(
    "symmetry, vector, expected",
    [
        (Ci, (1, 2, 3), [(1, 2, 3), (-1, -2, -3)]),
        (Csx, (1, 2, 3), [(1, 2, 3), (-1, 2, 3)]),
        (Csy, (1, 2, 3), [(1, 2, 3), (1, -2, 3)]),
        (Csz, (1, 2, 3), [(1, 2, 3), (1, 2, -3)]),
        (C2, (1, 2, 3), [(1, 2, 3), (-1, -2, 3)]),
        (C2v, (1, 2, 3), [(1, 2, 3), (1, -2, 3), (1, -2, -3), (1, 2, -3),]),
        (
            C4v,
            (1, 2, 3),
            [
                (1, 2, 3),
                (-2, 1, 3),
                (-1, -2, 3),
                (2, -1, 3),
                (-1, 2, 3),
                (2, 1, 3),
                (-2, -1, 3),
                (1, -2, 3),
            ],
        ),
        (
            D4,
            (1, 2, 3),
            [
                (1, 2, 3),
                (-2, 1, 3),
                (-1, -2, 3),
                (2, -1, 3),
                (-1, 2, -3),
                (2, 1, -3),
                (-2, -1, -3),
                (1, -2, -3),
            ],
        ),
        (
            C6,
            (1, 2, 3),
            [
                (1, 2, 3),
                (-1.232, 1.866, 3),
                (-2.232, -0.134, 3),
                (-1, -2, 3),
                (1.232, -1.866, 3),
                (2.232, 0.134, 3),
            ],
        ),
        (
            Td,
            (1, 2, 3),
            [
                (1, 2, 3),
                (3, 1, 2),
                (2, 3, 1),
                (-2, -1, 3),
                (3, -2, -1),
                (-1, 3, -2),
                (2, -1, -3),
                (-3, 2, -1),
                (-1, -3, 2),
                (1, -2, -3),
                (-3, 1, -2),
                (-2, -3, 1),
                (-1, -2, 3),
                (3, -1, -2),
                (-2, 3, -1),
                (2, 1, 3),
                (3, 2, 1),
                (1, 3, 2),
                (-2, 1, -3),
                (-3, -2, 1),
                (1, -3, -2),
                (-1, 2, -3),
                (-3, -1, 2),
                (2, -3, -1),
            ],
        ),
        (
            Oh,
            (1, 2, 3),
            [
                (1, 2, 3),
                (3, 1, 2),
                (2, 3, 1),
                (2, 1, -3),
                (-3, 2, 1),
                (1, -3, 2),
                (-2, 1, 3),
                (3, -2, 1),
                (1, 3, -2),
                (1, -2, -3),
                (-3, 1, -2),
                (-2, -3, 1),
                (-1, -2, 3),
                (3, -1, -2),
                (-2, 3, -1),
                (-2, -1, -3),
                (-3, -2, -1),
                (-1, -3, -2),
                (2, -1, 3),
                (3, 2, -1),
                (-1, 3, 2),
                (-1, 2, -3),
                (-3, -1, 2),
                (2, -3, -1),
                (-1, -2, -3),
                (-3, -1, -2),
                (-2, -3, -1),
                (-2, -1, 3),
                (3, -2, -1),
                (-1, 3, -2),
                (2, -1, -3),
                (-3, 2, -1),
                (-1, -3, 2),
                (-1, 2, 3),
                (3, -1, 2),
                (2, 3, -1),
                (1, 2, -3),
                (-3, 1, 2),
                (2, -3, 1),
                (2, 1, 3),
                (3, 2, 1),
                (1, 3, 2),
                (-2, 1, -3),
                (-3, -2, 1),
                (1, -3, -2),
                (1, -2, 3),
                (3, 1, -2),
                (-2, 3, 1),
            ],
        ),
    ],
    indirect=["vector"],
)
def test_symmetry(symmetry, vector, expected):
    vector_calculated = [
        tuple(v.round(3)) for v in symmetry.outer(vector).unique().data
    ]
    assert set(vector_calculated) == set(expected)


@pytest.mark.parametrize(
    "symmetry, expected",
    [(C2h, 4), (C6, 6), (D6h, 24), (T, 12), (Td, 24), (Oh, 48), (O, 24)],
)
def test_order(symmetry, expected):
    assert symmetry.order == expected


@pytest.mark.parametrize(
    "symmetry, expected", [(D2d, False), (C4, True), (C6v, False), (O, True),]
)
def test_is_proper(symmetry, expected):
    assert symmetry.is_proper == expected


@pytest.mark.parametrize(
    "symmetry, expected",
    [
        (C1, [C1]),
        (D2, [C1, C2x, C2y, C2z, D2]),
        (C6v, [C1, Csx, Csy, C2z, C3, C3v, C6, C6v]),
    ],
)
def test_subgroups(symmetry, expected):
    print(len(symmetry.subgroups))
    assert set(symmetry.subgroups) == set(expected)


@pytest.mark.parametrize(
    "symmetry, expected",
    [(C1, [C1]), (D2, [C1, C2x, C2y, C2z, D2]), (C6v, [C1, C2z, C3, C6]),],
)
def test_proper_subgroups(symmetry, expected):
    assert set(symmetry.proper_subgroups) == set(expected)


@pytest.mark.parametrize(
    "symmetry, expected",
    [
        (C1, C1),
        (Ci, C1),
        (C2, C2),
        (Cs, C1),
        (C2h, C2),
        (D2, D2),
        (C2v, C2x),
        (C4, C4),
        (C4h, C4),
        (C3h, C3),
        (C6v, C6),
        (D3h, D3y),
        (T, T),
        (Td, T),
        (Oh, O),
    ],
)
def test_proper_subgroup(symmetry, expected):
    assert symmetry.proper_subgroup._tuples == expected._tuples


@pytest.mark.parametrize(
    "symmetry, expected",
    [
        (C1, Ci),
        (Ci, Ci),
        (C2, C2h),
        (C2h, C2h),
        (C4, C4h),
        (C4h, C4h),
        (D4, D4h),
        (D4h, D4h),
        (C6v, D6h),
        (D6h, D6h),
        (T, Th),
        (Td, Oh),
    ],
)
def test_laue(symmetry, expected):
    assert symmetry.laue._tuples == expected._tuples


def test_is_laue():
    laue_groups = [Ci, C2h, D2h, C4h, D4h, S6, D3d, C6h, D6h, Th, Oh]
    assert all(i.contains_inversion for i in laue_groups)


@pytest.mark.parametrize(
    "symmetry, expected", [(Cs, C2), (C4v, D4), (Th, T), (Td, O), (O, O), (Oh, O),]
)
def test_proper_inversion_subgroup(symmetry, expected):
    assert symmetry.laue_proper_subgroup._tuples == expected._tuples


@pytest.mark.parametrize(
    "symmetry, expected",
    [
        (C1, False),
        (Ci, True),
        (Cs, False),
        (C2, False),
        (C2h, True),
        (D4, False),
        (D2d, False),
        (D3d, True),
        (C6, False),
        (C3h, False),
        (Td, False),
        (Oh, True),
    ],
)
def test_contains_inversion(symmetry, expected):
    assert symmetry.contains_inversion == expected


@pytest.mark.parametrize(
    "symmetry, other, expected",
    [(D2, C1, [C1]), (C1, C1, [C1]), (D2, C2, [C1, C2z]), (C4, S4, [C1, C2z]),],
)
def test_and(symmetry, other, expected):
    overlap = symmetry & other
    expected = Symmetry.from_generators(*expected)
    assert overlap._tuples == expected._tuples


@pytest.mark.parametrize(
    "symmetry, other, expected", [(C1, C1, True), (C1, C2, False),]
)
def test_eq(symmetry, other, expected):
    assert (symmetry == other) == expected


@pytest.mark.parametrize(
    "symmetry, expected",
    [
        (C1, np.zeros((0, 3))),
        (C2, [0, 1, 0]),
        (D2, [[0, 1, 0], [0, 0, 1]]),
        (C4, [[0, 1, 0], [1, 0, 0]]),
        (
            T,
            [
                [0.5 ** 0.5, -(0.5 ** 0.5), 0],
                [0, -(0.5 ** 0.5), 0.5 ** 0.5],
                [0, 0.5 ** 0.5, 0.5 ** 0.5],
                [0.5 ** 0.5, 0.5 ** 0.5, 0],
            ],
        ),
    ],
)
def test_fundamental_region(symmetry, expected):
    fr = symmetry.fundamental_region()
    assert np.allclose(fr.data, expected)


def test_no_symm_fundemental_sector():
    nosym = Symmetry.from_generators(Rotation([1, 0, 0, 0]))
    assert nosym.fundamental_region().size == 0


def test_get_point_group():
    """Makes sure all the ints from 1 to 230 give answers."""
    for sg_number in np.arange(1, 231):
        proper_pg = get_point_group(sg_number, proper=True)
        assert proper_pg in [C1, C2, C3, C4, C6, D2, D3, D4, D6, O, T]

        sg = GetSpaceGroup(sg_number)
        pg = get_point_group(sg_number, proper=False)
        assert proper_pg == spacegroup2pointgroup_dict[sg.point_group_name]["proper"]
        assert pg == spacegroup2pointgroup_dict[sg.point_group_name]["improper"]
