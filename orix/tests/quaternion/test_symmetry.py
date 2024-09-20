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

from copy import deepcopy

from diffpy.structure.spacegroups import GetSpaceGroup
from matplotlib import pyplot as plt
import numpy as np
import pytest

from orix.quaternion import Rotation, Symmetry, get_point_group

# fmt: off
# isort: off
from orix.quaternion.symmetry import (
    C1, Ci,  # triclinic
    C2x, C2y, C2z, Csx, Csy, Csz, Cs, C2, C2h,  # monoclinic
    D2, C2v, D2h,  # orthorhombic
    C4, S4, C4h, D4, C4v, D2d, D4h,  # tetragonal
    C3, S6, D3x, D3y, D3, C3v, D3d,  # trigonal
    C6, C3h, C6h, D6, C6v, D3h, D6h,  # hexagonal
    T, Th, O, Td, Oh,  # cubic
    spacegroup2pointgroup_dict, _groups, _get_unique_symmetry_elements
)
# isort: on
# fmt: on
from orix.vector import Vector3d


@pytest.fixture(params=[(1, 2, 3)])
def vector(request):
    return Vector3d(request.param)


@pytest.fixture(params=_groups)
def all_symmetries(request):
    return request.param


@pytest.mark.parametrize(
    "symmetry, vector, expected",
    [
        (Ci, (1, 2, 3), [(1, 2, 3), (-1, -2, -3)]),
        (Csx, (1, 2, 3), [(1, 2, 3), (-1, 2, 3)]),
        (Csy, (1, 2, 3), [(1, 2, 3), (1, -2, 3)]),
        (Csz, (1, 2, 3), [(1, 2, 3), (1, 2, -3)]),
        (C2, (1, 2, 3), [(1, 2, 3), (-1, -2, 3)]),
        (
            C2v,
            (1, 2, 3),
            [
                (1, 2, 3),
                (1, -2, 3),
                (1, -2, -3),
                (1, 2, -3),
            ],
        ),
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


def test_symmetry_repr():
    assert repr(Oh).split("\n")[0] == "Symmetry (48,) m-3m"


def test_same_symmetry_unique(all_symmetries):
    # test unique symmetry elements between two identical symmetries
    # are the symmetry itself
    symmetry = all_symmetries
    u = symmetry.outer(symmetry).unique()
    assert u.size == symmetry.size
    delta = (symmetry * ~u).angle
    assert np.allclose(delta, 0)
    assert np.allclose(u.data, symmetry.data)


def test_get_unique_symmetry_elements_symmetry_first_arg(all_symmetries):
    sym = all_symmetries
    assert sym in sym.subgroups
    result1 = []
    result2 = []
    for sg in sym.subgroups:
        # if 2nd arg is a subgroup of 1st arg then unique will be same
        # as symmetry
        u1 = _get_unique_symmetry_elements(sym, sg, check_subgroups=True)
        result1.append(u1)
        # explicit computation of sym1.outer(sym2).unique()
        u2 = _get_unique_symmetry_elements(sym, sg, check_subgroups=False)
        result2.append(u2)
    # in this case sym is explicitly returned by function
    assert all(s == sym for s in result1)
    # in this case sym is explicitly calculated by function
    assert all(s == sym for s in result2)


@pytest.mark.parametrize("symmetry", [C4, C4h, S4, D6, Th, O, Oh])
def test_get_unique_symmetry_elements_subgroup_first_arg(symmetry):
    sizes = []
    result = []
    for sg in symmetry.subgroups:
        u = _get_unique_symmetry_elements(sg, symmetry, check_subgroups=False)
        sizes.append(u.size == symmetry.size)
        result.append(u == symmetry)
    # sizes are the same
    assert all(sizes)
    # data is not the same as symmetry for all subgroups, order matters
    assert not all(result)


@pytest.mark.parametrize(
    "symmetry, expected",
    [(C2h, 4), (C6, 6), (D6h, 24), (T, 12), (Td, 24), (Oh, 48), (O, 24)],
)
def test_order(symmetry, expected):
    assert symmetry.order == expected


@pytest.mark.parametrize(
    "symmetry, expected",
    [
        (D2d, False),
        (C4, True),
        (C6v, False),
        (O, True),
    ],
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
    assert set(symmetry.subgroups) == set(expected)


@pytest.mark.parametrize(
    "symmetry, expected",
    [
        (C1, [C1]),
        (D2, [C1, C2x, C2y, C2z, D2]),
        (C6v, [C1, C2z, C3, C6]),
    ],
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
    "symmetry, expected",
    [
        (Cs, C2),
        (C4v, D4),
        (Th, T),
        (Td, O),
        (O, O),
        (Oh, O),
    ],
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
    [
        (D2, C1, [C1]),
        (C1, C1, [C1]),
        (D2, C2, [C1, C2z]),
        (C4, S4, [C1, C2z]),
    ],
)
def test_and(symmetry, other, expected):
    overlap = symmetry & other
    expected = Symmetry.from_generators(*expected)
    assert overlap._tuples == expected._tuples


@pytest.mark.parametrize(
    "symmetry, other, expected",
    [
        (C1, C1, True),
        (C1, C2, False),
    ],
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
                [0.5**0.5, -(0.5**0.5), 0],
                [0, -(0.5**0.5), 0.5**0.5],
                [0, 0.5**0.5, 0.5**0.5],
                [0.5**0.5, 0.5**0.5, 0],
            ],
        ),
    ],
)
def test_fundamental_zone(symmetry, expected):
    fz = symmetry.fundamental_zone()
    assert np.allclose(fz.data, expected)


def test_no_symm_fundamental_zone():
    nosym = Symmetry.from_generators(Rotation([1, 0, 0, 0]))
    assert nosym.fundamental_zone().size == 0


def test_get_point_group():
    """Makes sure all the ints from 1 to 230 give answers."""
    for sg_number in np.arange(1, 231):
        proper_pg = get_point_group(sg_number, proper=True)
        assert proper_pg in [C1, C2, C3, C4, C6, D2, D3, D4, D6, O, T]

        sg = GetSpaceGroup(sg_number)
        pg = get_point_group(sg_number, proper=False)
        assert proper_pg == spacegroup2pointgroup_dict[sg.point_group_name]["proper"]
        assert pg == spacegroup2pointgroup_dict[sg.point_group_name]["improper"]


def test_unique_symmetry_elements_subgroups(all_symmetries):
    # test that the unique symmetry elements between a symmetry and its
    # subgroups are the original symmetry
    sym = all_symmetries
    for sg in sym.subgroups:
        # outer of symmetry with its subgroups
        u = sym.outer(sg).unique()
        # assert that unique is same size as main symmetry
        assert u.size == sym.size
        # check that there is no difference between unique
        # and main symmetry
        assert np.allclose((sym * ~u).angle, 0)


def test_two_symmetries_are_not_in_each_others_subgroup(all_symmetries):
    # if given two symmetries, test that both do not exist in the
    # subgroup of the other
    sym1 = all_symmetries
    # identify place in list by name, cannot test symmetry directy as D3
    # and D3x are the same and causes an index issue
    i = [s.name for s in _groups].index(sym1.name)
    if i + 1 < len(_groups):
        values = []
        # only test successive symmetries in _groups to avoid repetition
        for sym2 in _groups[i + 1 :]:
            if {sym1.name, sym2.name} == {"32", "321"}:
                # D3 and D3x are defined to be the same, so do not test
                continue
            sym2_in_sym1_sg = True if sym2 in sym1.subgroups else False
            sym1_in_sym2_sg = True if sym1 in sym2.subgroups else False
            values.append(sym2_in_sym1_sg + sym1_in_sym2_sg)
        # value==0 is okay, ie. unrelated symmetries
        # value==1 is okay, ie. only one is subgroup of other
        # if value==2 then both symmetries exist in subgroup of other
        assert not any(v == 2 for v in values)


def test_unique_unrelated_symmetries():
    sym1 = D6
    sym2 = C4
    assert sym1 not in sym2.subgroups
    assert sym2 not in sym1.subgroups
    # unique will be computed manually
    sym12 = _get_unique_symmetry_elements(sym1, sym2)
    sym21 = _get_unique_symmetry_elements(sym2, sym1)
    sym12 = sym12[np.lexsort(sym12.data.T)]
    sym21 = sym21[np.lexsort(sym21.data.T)]
    assert sym12.size == sym21.size
    # symmetry order matters, as discussed in
    # DOI: http://dx.doi.org/10.1098/rspa.2017.0274
    assert not np.allclose(sym12.data, sym21.data)


def test_hash():
    h = [hash(s) for s in _groups]
    assert len(set(h)) == len(_groups)


def test_hash_persistence():
    h1 = [hash(s) for s in _groups]
    h2 = [hash(deepcopy(s)) for s in _groups]
    assert all(h1a == h2a for h1a, h2a in zip(h1, h2))


@pytest.mark.parametrize("pg", [C1, C4, Oh])
def test_symmetry_plot(pg):
    fig = pg.plot(return_figure=True)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]

    c0 = ax.collections[0]
    assert len(c0.get_offsets()) == np.count_nonzero(~pg.improper)
    assert c0.get_label().lower() == "upper"
    if not pg.is_proper:
        c1 = ax.collections[1]
        assert len(c1.get_offsets()) == np.count_nonzero(pg.improper)
        assert c1.get_label().lower() == "lower"

    assert len(ax.texts) == 2
    assert ax.texts[0].get_text() == "$e_1$"
    assert ax.texts[1].get_text() == "$e_2$"

    plt.close("all")


@pytest.mark.parametrize("symmetry", [C1, C4, Oh])
def test_symmetry_plot_raises(symmetry):
    with pytest.raises(TypeError, match="Orientation must be a Rotation instance"):
        _ = symmetry.plot(return_figure=True, orientation="test")


class TestFundamentalSectorFromSymmetry:
    """Test the normals, vertices and centers of the fundamental sector
    for all 32 crystallographic point groups.
    """

    def test_fundamental_sector_c1(self):
        pg = C1  # 1
        fs = pg.fundamental_sector
        assert fs.data.size == 0
        assert fs.vertices.data.size == 0
        assert fs.center.data.size == 0
        assert fs.edges.data.size == 0

    def test_fundamental_sector_ci(self):
        pg = Ci  # -1
        fs = pg.fundamental_sector
        normal = [[0, 0, 1]]
        assert np.allclose(fs.data, normal)
        assert fs.vertices.data.size == 0
        assert np.allclose(fs.center.data, normal)

    def test_fundamental_sector_c2(self):
        pg = C2  # 2
        fs = pg.fundamental_sector
        normal = [[0, 1, 0]]
        assert np.allclose(fs.data, normal)
        assert fs.vertices.data.size == 0
        assert np.allclose(fs.center.data, normal)

    def test_fundamental_sector_cs(self):
        pg = Cs  # m
        fs = pg.fundamental_sector
        normal = [[0, 0, 1]]
        assert np.allclose(fs.data, normal)
        assert fs.vertices.data.size == 0
        assert np.allclose(fs.center.data, normal)

    def test_fundamental_sector_c2h(self):
        pg = C2h  # 2/m
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0]])
        assert np.allclose(fs.vertices.data, [[1, 0, 0], [-1, 0, 0]])
        assert np.allclose(fs.center.data, [[0, 0.5, 0.5]])

    def test_fundamental_sector_d2(self):
        pg = D2  # 222
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0]])
        assert np.allclose(fs.vertices.data, [[1, 0, 0], [-1, 0, 0]])
        assert np.allclose(fs.center.data, [[0, 0.5, 0.5]])

    def test_fundamental_sector_c2v(self):
        pg = C2v  # mm2
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0]])
        assert np.allclose(fs.vertices.data, [[1, 0, 0], [-1, 0, 0]])
        assert np.allclose(fs.center.data, [[0, 0.5, 0.5]])

    def test_fundamental_sector_d2h(self):
        pg = D2h  # mmm
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        assert np.allclose(fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert np.allclose(fs.center.data, [[1 / 3, 1 / 3, 1 / 3]])

    def test_fundamental_sector_c4(self):
        pg = C4  # 4
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 1, 0], [1, 0, 0]])
        assert np.allclose(fs.vertices.data, [[0, 0, 1], [0, 0, -1]])
        assert np.allclose(fs.center.data, [[0.5, 0.5, 0]])

    def test_fundamental_sector_s4(self):
        pg = S4  # -4
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0]])
        assert np.allclose(fs.vertices.data, [[1, 0, 0], [-1, 0, 0]])
        assert np.allclose(fs.center.data, [[0, 0.5, 0.5]])

    def test_fundamental_sector_c4h(self):
        pg = C4h  # 4/m
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        assert np.allclose(fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert np.allclose(fs.center.data, [[1 / 3, 1 / 3, 1 / 3]])

    def test_fundamental_sector_d4(self):
        pg = D4  # 422
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        assert np.allclose(fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        assert np.allclose(fs.center.data, [[1 / 3, 1 / 3, 1 / 3]])

    def test_fundamental_sector_c4v(self):
        pg = C4v  # 4mm
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 1, 0], [0.7071, -0.7071, 0]], atol=1e-4)
        assert np.allclose(fs.vertices.data, [[0, 0, 1], [0, 0, -1]])
        assert np.allclose(fs.center.data, [[0.3536, 0.1464, 0]], atol=1e-4)

    def test_fundamental_sector_d2d(self):
        pg = D2d  # -42m
        fs = pg.fundamental_sector
        assert np.allclose(
            fs.data, [[0, 0, 1], [0.7071, 0.7071, 0], [0.7071, -0.7071, 0]], atol=1e-4
        )
        assert np.allclose(
            fs.vertices.data, [[0.7071, -0.7071, 0], [0, 0, 1], [0.7071, 0.7071, 0]]
        )
        assert np.allclose(fs.center.data, [[0.4714, 0, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_d4h(self):
        pg = D4h  # 4/mmm
        fs = pg.fundamental_sector
        assert np.allclose(
            fs.data, [[0, 0, 1], [0, 1, 0], [0.7071, -0.7071, 0]], atol=1e-4
        )
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0.7071, 0.7071, 0]], atol=1e-4
        )
        assert np.allclose(fs.center.data, [[0.569, 0.2357, 1 / 3]], atol=1e-3)

    def test_fundamental_sector_c3(self):
        pg = C3  # 3
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 1, 0], [0.866, 0.5, 0]], atol=1e-3)
        assert np.allclose(fs.vertices.data, [[0, 0, 1], [0, 0, -1]])
        assert np.allclose(fs.center.data, [[0.433, 0.75, 0]], atol=1e-4)

    def test_fundamental_sector_s6(self):
        pg = S6  # -3
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [0.866, 0.5, 0]], atol=1e-3)
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [-0.5, 0.866, 0]], atol=1e-4
        )
        assert np.allclose(fs.center.data, [[1 / 6, 0.2887, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_d3(self):
        pg = D3  # 32
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [0.866, 0.5, 0]], atol=1e-3)
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [-0.5, 0.866, 0]], atol=1e-4
        )
        assert np.allclose(fs.center.data, [[1 / 6, 0.2887, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_c3v(self):
        pg = C3v  # 3m
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0.5, 0.866, 0], [0.5, -0.866, 0]], atol=1e-3)
        assert np.allclose(fs.vertices.data, [[0, 0, 1], [0, 0, -1]])
        assert np.allclose(fs.center.data, [[0.5, 0, 0]])

    def test_fundamental_sector_d3d(self):
        pg = D3d  # -3m
        fs = pg.fundamental_sector
        assert np.allclose(
            fs.data, [[0, 0, 1], [0.5, 0.866, 0], [0.5, -0.866, 0]], atol=1e-3
        )
        assert np.allclose(
            fs.vertices.data, [[0.866, -0.5, 0], [0, 0, 1], [0.866, 0.5, 0]], atol=1e-3
        )
        assert np.allclose(fs.center.data, [[0.577, 0, 1 / 3]], atol=1e-3)

    def test_fundamental_sector_c6(self):
        pg = C6  # 6
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 1, 0], [0.866, -0.5, 0]], atol=1e-3)
        assert np.allclose(fs.vertices.data, [[0, 0, 1], [0, 0, -1]])
        assert np.allclose(fs.center.data, [[0.433, 0.25, 0]], atol=1e-3)

    def test_fundamental_sector_c3h(self):
        pg = C3h  # -6
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [0.866, 0.5, 0]], atol=1e-3)
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [-0.5, 0.866, 0]], atol=1e-3
        )
        assert np.allclose(fs.center.data, [[1 / 6, 0.2887, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_c6h(self):
        pg = C6h  # 6/m
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [0.866, -0.5, 0]], atol=1e-3)
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0.5, 0.866, 0]], atol=1e-3
        )
        assert np.allclose(fs.center.data, [[0.5, 0.2887, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_d6(self):
        pg = D6  # 622
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [0.866, -0.5, 0]], atol=1e-3)
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0.5, 0.866, 0]], atol=1e-3
        )
        assert np.allclose(fs.center.data, [[0.5, 0.2887, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_c6v(self):
        pg = C6v  # 6mm
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 1, 0], [0.5, -0.866, 0]], atol=1e-3)
        assert np.allclose(fs.vertices.data, [[0, 0, 1], [0, 0, -1]])
        assert np.allclose(fs.center.data, [[0.25, 0.067, 0]], atol=1e-3)

    def test_fundamental_sector_d3h(self):
        pg = D3h  # -6m2
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [0.866, -0.5, 0]], atol=1e-3)
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0.5, 0.866, 0]], atol=1e-3
        )
        assert np.allclose(fs.center.data, [[0.5, 0.2887, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_d6h(self):
        pg = D6h  # 6/mmm
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[0, 0, 1], [0, 1, 0], [0.5, -0.866, 0]], atol=1e-3)
        assert np.allclose(
            fs.vertices.data, [[1, 0, 0], [0, 0, 1], [0.866, 0.5, 0]], atol=1e-3
        )
        assert np.allclose(fs.center.data, [[0.622, 0.1667, 1 / 3]], atol=1e-4)

    def test_fundamental_sector_t(self):
        pg = T  # 23
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[1, 1, 0], [1, -1, 0], [0, -1, 1], [0, 1, 1]])
        assert np.allclose(
            fs.vertices.data,
            [[0, 0, 1], [0.5774, 0.5774, 0.5774], [1, 0, 0], [0.5774, -0.5774, 0.5774]],
            atol=1e-4,
        )
        assert np.allclose(fs.center.data, [[0.7076, -0.0004, 0.7067]], atol=1e-4)

    def test_fundamental_sector_th(self):
        pg = Th  # m-3
        fs = pg.fundamental_sector
        assert np.allclose(
            fs.data,
            [[1, 0, 0], [0, -1, 1], [-1, 0, 1], [0, 1, 0], [0, 0, 1]],
        )
        assert np.allclose(
            fs.vertices.data,
            [
                [0, 0.7071, 0.7071],
                [0.5774, 0.5774, 0.5774],
                [0.7071, 0, 0.7071],
                [0, 0, 1],
            ],
            atol=1e-3,
        )
        assert np.allclose(fs.center.data, [[0.3499, 0.3481, 0.8697]], atol=1e-4)

    def test_fundamental_sector_o(self):
        pg = O  # 432
        fs = pg.fundamental_sector
        assert np.allclose(
            fs.data, [[1, 0, 0], [0, -1, 1], [-1, 0, 1], [0, 1, 0], [0, 0, 1]]
        )
        assert np.allclose(
            fs.vertices.data,
            [
                [0, 0.7071, 0.7071],
                [0.5774, 0.5774, 0.5774],
                [0.7071, 0, 0.7071],
                [0, 0, 1],
            ],
            atol=1e-3,
        )
        assert np.allclose(fs.center.data, [[0.3499, 0.3481, 0.8697]], atol=1e-4)

    def test_fundamental_sector_td(self):
        pg = Td  # -43m
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[1, -1, 0], [1, 1, 0], [-1, 0, 1]])
        assert np.allclose(
            fs.vertices.data,
            [[0.5774, 0.5774, 0.5774], [0, 0, 1], [0.5774, -0.5774, 0.5774]],
            atol=1e-3,
        )
        assert np.allclose(fs.center.data, [[0.3849, 0, 0.7182]], atol=1e-4)

    def test_fundamental_sector_oh(self):
        pg = Oh  # m-3m
        fs = pg.fundamental_sector
        assert np.allclose(fs.data, [[1, -1, 0], [-1, 0, 1], [0, 1, 0]])
        assert np.allclose(
            fs.vertices.data,
            [[0.5774, 0.5774, 0.5774], [0.7071, 0, 0.7071], [0, 0, 1]],
            atol=1e-4,
        )
        assert np.allclose(fs.center.data, [[0.4282, 0.1925, 0.7615]], atol=1e-4)

    # ---------- End of the 32 crystallographic point groups --------- #

    def test_fundamental_sector_c2x(self):
        pg = C2x  # 211
        fs = pg.fundamental_sector
        normal = [[0, 0, 1]]
        assert np.allclose(fs.data, normal)
        assert np.allclose(fs.vertices.data, np.zeros((0, 3)))
        assert np.allclose(fs.center.data, normal)

    def test_fundamental_sector_csx(self):
        pg = Csx  # m11
        fs = pg.fundamental_sector
        normal = [[0, 0, -1]]
        assert np.allclose(fs.data, normal)
        assert np.allclose(fs.vertices.data, np.zeros((0, 3)))
        assert np.allclose(fs.center.data, normal)


@pytest.mark.parametrize("symmetry", [C1, C2, C4, D6, T, Oh])
def test_equality(symmetry):
    # test that inherited equality is properly tested
    assert Rotation(symmetry) == symmetry


class TestLaueGroup:
    def test_crystal_system(self):
        assert Ci.system == "triclinic"
        assert C2h.system == "monoclinic"
        assert D2h.system == "orthorhombic"
        assert D4h.system == "tetragonal"
        assert D3d.system == "trigonal"
        assert D6h.system == "hexagonal"
        assert Oh.system == "cubic"
        assert Symmetry(((1, 0, 0, 0), (1, 1, 0, 0))).system is None

    def test_laue_group_name(self):
        assert Ci.laue.name == "-1"
        assert C2h.laue.name == "2/m"
        assert D2h.laue.name == "mmm"
        assert C4h.laue.name == "4/m"
        assert D4h.laue.name == "4/mmm"
        assert S6.laue.name == "-3"
        assert D3d.laue.name == "-3m"
        assert C6h.laue.name == "6/m"
        assert D6h.laue.name == "6/mmm"
        assert Th.laue.name == "m-3"
        assert Oh.laue.name == "m-3m"
        assert Symmetry(((1, 0, 0, 0), (1, 1, 0, 0))).laue.name is None


class TestEulerFundamentalRegion:
    """Test functionality used in
    :meth:`~orix.quaternion.Orientation.in_euler_fundamental_region`.
    """

    def test_euler_fundamental_region(self):
        # Proper subgroups
        # fmt: off
        assert np.allclose(C1.euler_fundamental_region,  (360, 180, 360))
        assert np.allclose(C2x.euler_fundamental_region, (360,  90, 360))
        assert np.allclose(C2y.euler_fundamental_region, (360,  90, 360))
        assert np.allclose(C2z.euler_fundamental_region, (360, 180, 180))
        assert np.allclose(D2.euler_fundamental_region,  (360,  90, 180))
        assert np.allclose(C4.euler_fundamental_region,  (360, 180,  90))
        assert np.allclose(D4.euler_fundamental_region,  (360,  90,  90))
        assert np.allclose(C3.euler_fundamental_region,  (360, 180, 120))
        assert np.allclose(D3.euler_fundamental_region,  (360,  90, 120))
        assert np.allclose(D3y.euler_fundamental_region, (360,  90, 120))
        assert np.allclose(C6.euler_fundamental_region,  (360, 180,  60))
        assert np.allclose(D6.euler_fundamental_region,  (360,  90,  60))
        assert np.allclose(T.euler_fundamental_region,   (360,  90, 180))
        assert np.allclose(O.euler_fundamental_region,   (360,  90, 90))
        # fmt: on

        # Unknown symmetry
        unrecognized_symmetry = Symmetry.random(4)
        assert np.allclose(
            unrecognized_symmetry.euler_fundamental_region, (360, 180, 360)
        )

        # All point groups provide a region
        for pg in _groups:
            angles = pg.euler_fundamental_region
            if pg.name in ["1", "-1", "2", "m11", "1m1", "11m"]:
                assert np.allclose(angles, (360, 180, 360))
            else:
                assert not np.allclose(angles, (360, 180, 360))

    def test_primary_axis_order(self):
        for pg in [C1, C2x, C2y]:
            assert pg._primary_axis_order == 1
        for pg in [C2z, D2, T]:
            assert pg._primary_axis_order == 2
        for pg in [C3, D3x, D3y, D3]:
            assert pg._primary_axis_order == 3
        for pg in [C4, D4, Oh]:
            assert pg._primary_axis_order == 4
        for pg in [C6, D6]:
            assert pg._primary_axis_order == 6

        unrecognized_symmetry = Symmetry.random(4)
        assert unrecognized_symmetry._primary_axis_order is None

        # All point groups provide an order
        for pg in _groups:
            assert pg._primary_axis_order != 0

    def test_special_rotation(self):
        for pg in [C1, C2z, C3, C4, C6]:
            assert np.allclose(pg._special_rotation.data, (1, 0, 0, 0))
        assert np.allclose(C2x._special_rotation.data, ((1, 0, 0, 0), (0, 1, 0, 0)))
        assert np.allclose(C2y._special_rotation.data, ((1, 0, 0, 0), (0, 0, 1, 0)))
        for pg in [D2, D4, D6, D3]:
            assert np.allclose(pg._special_rotation.data, ((1, 0, 0, 0), (0, -1, 0, 0)))
        assert np.allclose(
            D3y._special_rotation.data,
            ((1, 0, 0, 0), (0, -1 / np.sqrt(2), 1 / np.sqrt(2), 0)),
        )
        assert np.allclose(
            T._special_rotation.data,
            (
                (1, 0, 0, 0),
                (0.5, -0.5, -0.5, -0.5),
                (-0.5, -0.5, -0.5, -0.5),
                (0, -1, 0, 0),
                (-0.5, -0.5, 0.5, -0.5),
                (-0.5, 0.5, 0.5, -0.5),
            ),
        )
        assert np.allclose(
            O._special_rotation.data,
            (
                (1, 0, 0, 0),
                (0.5, -0.5, -0.5, -0.5),
                (-0.5, -0.5, -0.5, -0.5),
                (0, -1 / np.sqrt(2), -1 / np.sqrt(2), 0),
                (-1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0),
                (-1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0),
            ),
        )

        unrecognized_symmetry = Symmetry.random(4)
        assert np.allclose(unrecognized_symmetry._special_rotation.data, (1, 0, 0, 0))

        # All point groups provide at least one rotation
        for pg in _groups:
            assert isinstance(pg._special_rotation.data, np.ndarray)
