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

from diffpy.structure import Lattice, Structure
from diffpy.structure.spacegroups import GetSpaceGroup
import numpy as np
import pytest

from orix.crystal_map import Phase, PhaseList


class TestPhaseList:
    @pytest.mark.parametrize("empty_input", [(), [], {}])
    def test_init_empty_phaselist(self, empty_input):
        pl = PhaseList(empty_input)
        assert repr(pl) == "No phases."
        pl.add(Phase("al", point_group="m-3m"))
        assert repr(pl) == (
            "Id  Name  Space group  Point group  Proper point group     Color\n"
            " 0    al         None         m-3m                 432  tab:blue"
        )

    def test_init_set_to_nones(self):
        phase_ids = [1, 2]
        pl = PhaseList(ids=phase_ids)

        assert pl.ids == phase_ids
        assert pl.names == [""] * 2
        assert pl.point_groups == [None] * 2
        assert pl.space_groups == [None] * 2
        assert pl.colors == ["tab:blue", "tab:orange"]
        assert pl.structures == [Structure()] * 2

    @pytest.mark.parametrize("phase_collection", ["dict", "list"])
    def test_init_phaselist_from_phases(self, phase_collection):
        p1 = Phase(name="austenite", point_group=432, color=None)
        p2 = Phase(name="ferrite", point_group="432", color="C1")
        if phase_collection == "dict":
            phases = {1: p1, 2: p2}
        else:  # phase_collection == "list":
            phases = [p1, p2]

        pl = PhaseList(phases)

        assert pl.names == [p.name for p in [p1, p2]]
        assert pl.point_groups == [p.point_group for p in [p1, p2]]
        assert pl.space_groups == [p.space_group for p in [p1, p2]]
        assert pl.colors == [p.color for p in [p1, p2]]
        assert pl.colors_rgb == [p.color_rgb for p in [p1, p2]]

    def test_init_phaselist_from_phase(self):
        p = Phase(name="austenite", point_group="432", color="C2")
        pl = PhaseList(p)

        assert pl.names == [p.name]
        assert pl.point_groups == [p.point_group]
        assert pl.space_groups == [p.space_group]
        assert pl.colors == [p.color]
        assert pl.colors_rgb == [p.color_rgb]

    @pytest.mark.parametrize(
        (
            "names, space_groups, point_groups, colors, phase_ids, desired_names, "
            "desired_space_groups, desired_point_groups, desired_colors, "
            "desired_phase_ids"
        ),
        [
            (
                ["al", "ni"],
                [210],
                [43],
                [None, "C1"],
                [1],
                ["al", "ni"],
                ["F4132", None],
                ["432", None],
                ["tab:blue", "tab:orange"],
                [1, 2],
            ),
            (
                ["al", None],
                [210, 225],
                [432, "m3m"],
                (1, 0, 0),
                [100],
                ["al", ""],
                ["F4132", "Fm-3m"],
                ["432", "m-3m"],
                ["r", "tab:blue"],
                [100, 101],
            ),
            (
                [None],
                [None, None],
                [None, None],
                ["green", "black"],
                1,
                ["", ""],
                [None, None],
                [None, None],
                ["g", "k"],
                [1, 2],
            ),
            (
                ["al", "Ni"],
                [225, 145, None],
                ["m-3m", 3, None],
                ["C0", None, "C0"],
                None,
                ["al", "Ni", ""],
                ["Fm-3m", "P32", None],
                ["m-3m", "3", None],
                ["tab:blue", "tab:orange", "tab:blue"],
                [0, 1, 2],
            ),
            ("al", 210, 43, "C0", [0], ["al"], ["F4132"], ["432"], ["tab:blue"], [0]),
        ],
    )
    def test_init_phaselist_from_strings(
        self,
        names,
        space_groups,
        point_groups,
        colors,
        phase_ids,
        desired_names,
        desired_space_groups,
        desired_point_groups,
        desired_colors,
        desired_phase_ids,
    ):
        pl = PhaseList(
            names=names,
            space_groups=space_groups,
            point_groups=point_groups,
            colors=colors,
            ids=phase_ids,
        )

        actual_point_group_names = []
        actual_space_group_names = []
        for _, p in pl:
            if p.point_group is None:
                actual_point_group_names.append(None)
            else:
                actual_point_group_names.append(p.point_group.name)
            if p.space_group is None:
                actual_space_group_names.append(None)
            else:
                actual_space_group_names.append(p.space_group.short_name)

        assert pl.names == desired_names
        assert actual_space_group_names == desired_space_groups
        assert actual_point_group_names == desired_point_groups
        assert pl.colors == desired_colors
        assert pl.ids == desired_phase_ids

    def test_init_with_single_structure(self):
        structure = Structure()
        names = ["a", "b"]
        pl = PhaseList(names=names, structures=structure)

        assert pl.names == names
        assert pl.structures == [structure] * 2

    def test_get_phaselist_colors_rgb(self):
        pl = PhaseList(names=["a", "b", "c"], colors=["r", "g", (0, 0, 1)])

        assert pl.colors == ["r", "g", "b"]
        assert np.allclose(pl.colors_rgb, [(1.0, 0.0, 0.0), [0, 0.5, 0], (0, 0, 1)])

    @pytest.mark.parametrize("n_names", [1, 3])
    def test_get_phaselist_size(self, n_names):
        phase_names_pool = "abcd"
        phase_names = [phase_names_pool[i] for i in range(n_names)]

        pl = PhaseList(names=phase_names)

        assert pl.size == n_names

    @pytest.mark.parametrize(
        "n_names, phase_ids, desired_names, desired_phase_ids",
        [
            (2, [0, 2], ["a", "b"], [0, 2]),
            (3, [1, 100, 2], ["a", "c", "b"], [1, 2, 100]),
            (3, 100, ["a", "b", "c"], [100, 101, 102]),
        ],
    )
    def test_get_phaselist_ids(
        self, n_names, phase_ids, desired_names, desired_phase_ids
    ):
        phase_names_pool = "abc"
        phase_names = [phase_names_pool[i] for i in range(n_names)]

        pl = PhaseList(names=phase_names, ids=phase_ids)

        assert pl.names == desired_names
        assert pl.ids == desired_phase_ids

    @pytest.mark.parametrize(
        (
            "key_getter, desired_name, desired_space_group, desired_point_group, "
            "desired_proper_point_group, desired_color"
        ),
        [
            (0, "a", "Im-3m", "m-3m", "432", "r"),
            ("b", "b", "P432", "432", "432", "g"),
            (slice(2, None, None), "c", "P3", "3", "3", "b"),  # equivalent to pl[2:]
        ],
    )
    def test_get_phase_from_phaselist(
        self,
        phase_list,
        key_getter,
        desired_name,
        desired_space_group,
        desired_point_group,
        desired_proper_point_group,
        desired_color,
    ):
        p = phase_list[key_getter]

        assert repr(p) == (
            f"<name: {desired_name}. space group: {desired_space_group}. point group: "
            f"{desired_point_group}. proper point group: {desired_proper_point_group}. "
            f"color: {desired_color}>"
        )

    @pytest.mark.parametrize(
        "key_getter, desired_names, desired_point_groups, desired_colors",
        [
            (
                slice(0, None, None),
                ["a", "b", "c"],
                ["m-3m", "432", "3"],
                ["r", "g", "b"],
            ),
            (("a", "b"), ["a", "b"], ["m-3m", "432"], ["r", "g"]),
            (["a", "b"], ["a", "b"], ["m-3m", "432"], ["r", "g"]),
            ((0, 2), ["a", "c"], ["m-3m", "3"], ["r", "b"]),
            ([0, 2], ["a", "c"], ["m-3m", "3"], ["r", "b"]),
        ],
    )
    def test_get_phases_from_phaselist(
        self,
        phase_list,
        key_getter,
        desired_names,
        desired_point_groups,
        desired_colors,
    ):
        phases = phase_list[key_getter]

        assert phases.names == desired_names
        assert [p.name for p in phases.point_groups] == desired_point_groups
        assert phases.colors == desired_colors

    @pytest.mark.parametrize("key_getter", ["d", 3, slice(3, None)])
    def test_get_from_phaselist_error(self, phase_list, key_getter):
        with pytest.raises(KeyError):
            _ = phase_list[key_getter]

    @pytest.mark.parametrize(
        "add_not_indexed, desired_ids", [(True, [-1, 0, 1]), (False, [0, 1, 2])]
    )
    def test_get_from_phaselist_not_indexed(
        self, phase_list, add_not_indexed, desired_ids
    ):
        if add_not_indexed:
            phase_list.add_not_indexed()
        assert phase_list[:3].ids == desired_ids

    def test_add_phase_in_empty_phaselist(self):
        """Add Phase to empty PhaseList."""
        sg_no = 10
        name = "a"
        pl = PhaseList()
        pl.add(Phase(name, space_group=sg_no))
        assert pl.ids == [0]
        assert pl.names == [name]
        assert pl.space_groups == [GetSpaceGroup(sg_no)]
        assert pl.structures == [Structure()]

    def test_add_list_phases_to_phaselist(self):
        """Add a list of Phase objects to PhaseList, also ensuring that
        unique colors are given.
        """
        names = ["a", "b"]
        sg_no = [10, 20]
        colors = ["tab:blue", "tab:orange"]
        pl = PhaseList(names=names, space_groups=sg_no)
        assert pl.colors == colors

        new_names = ["c", "d"]
        new_sg_no = [30, 40]
        pl.add([Phase(name=n, space_group=i) for n, i in zip(new_names, new_sg_no)])
        assert pl.names == names + new_names
        assert pl.space_groups == (
            [GetSpaceGroup(i) for i in sg_no] + [GetSpaceGroup(i) for i in new_sg_no]
        )
        assert pl.colors == colors + ["tab:green", "tab:red"]

    def test_add_phaselist_to_phaselist(self):
        """Add a PhaseList to a PhaseList, also ensuring that new IDs are
        given.
        """
        names = ["a", "b"]
        sg_no = [10, 20]
        pl1 = PhaseList(names=names, space_groups=sg_no)
        assert pl1.ids == [0, 1]

        names2 = ["c", "d"]
        sg_no2 = [30, 40]
        ids = [4, 5]
        pl2 = PhaseList(names=names2, space_groups=sg_no2, ids=ids)
        pl1.add(pl2)
        assert pl1.names == names + names2
        assert pl1.space_groups == (
            [GetSpaceGroup(i) for i in sg_no] + [GetSpaceGroup(i) for i in sg_no2]
        )
        assert pl1.ids == [0, 1, 2, 3]

    def test_add_to_phaselist_raises(self):
        """Trying to add a Phase with a name already in the PhaseList
        raises a ValueError.
        """
        pl = PhaseList(names=["a"])
        with pytest.raises(ValueError, match="'a' is already in the phase list"):
            pl.add(Phase("a"))

    @pytest.mark.parametrize(
        "key_del, invalid_phase, error_type, error_msg",
        [
            (0, False, None, None),
            ("a", False, None, None),
            (3, True, KeyError, "3"),
            ("d", True, KeyError, "d is not among the phase names"),
            ([0, 1], True, TypeError, ".* is an invalid phase ID or"),
        ],
    )
    def test_del_phase_in_phaselist(
        self, phase_list, key_del, invalid_phase, error_type, error_msg
    ):
        if invalid_phase:
            with pytest.raises(error_type, match=error_msg):
                del phase_list[key_del]
        else:
            phase_ids = phase_list.ids
            names = phase_list.names

            del phase_list[key_del]

            if isinstance(key_del, int):
                phase_ids.remove(key_del)
                assert phase_list.ids == phase_ids
            elif isinstance(key_del, str):
                names.remove(key_del)
                assert phase_list.names == names

    def test_iterate_phaselist(self):
        names = ["al", "ni", "sigma"]
        point_groups = [3, 432, "m-3m"]
        colors = ["g", "b", "r"]
        structures = [
            Structure(),
            Structure(lattice=Lattice(1, 2, 3, 90, 90, 90)),
            Structure(),
        ]

        pl = PhaseList(
            names=names, point_groups=point_groups, colors=colors, structures=structures
        )

        for i, ((phase_id, phase), n, s, c, structure) in enumerate(
            zip(pl, names, point_groups, colors, structures)
        ):
            assert phase_id == i
            assert phase.name == n
            assert phase.point_group.name == str(s)
            assert phase.color == c
            assert phase.structure == structure

    def test_deepcopy_phaselist(self, phase_list):
        names = phase_list.names
        point_groups = [s.name for s in phase_list.point_groups]
        colors = phase_list.colors

        pl2 = phase_list.deepcopy()
        assert pl2.names == names

        phase_list.add(Phase("d", point_group="m-3m"))
        phase_list["d"].color = "g"

        assert phase_list.names == names + ["d"]
        assert [s.name for s in phase_list.point_groups] == point_groups + ["m-3m"]
        assert phase_list.colors == colors + ["g"]

        assert pl2.names == names
        assert [s.name for s in pl2.point_groups] == point_groups
        assert pl2.colors == colors

    def test_shallowcopy_phaselist(self, phase_list):
        pl2 = phase_list

        phase_list.add(Phase("d", point_group="m-3m"))

        assert pl2.names == phase_list.names
        assert [s2.name for s2 in pl2.point_groups] == [
            s.name for s in phase_list.point_groups
        ]
        assert pl2.colors == phase_list.colors

    def test_make_not_indexed(self):
        phase_names = ["a", "b", "c"]
        phase_colors = ["r", "g", "b"]
        pl = PhaseList(names=phase_names, colors=phase_colors, ids=[-1, 0, 1])

        assert pl.names == phase_names
        assert pl.colors == phase_colors

        pl.add_not_indexed()

        phase_names[0] = "not_indexed"
        phase_colors[0] = "w"
        assert pl.names == phase_names
        assert pl.colors == phase_colors

    def test_phase_id_from_name(self, phase_list):
        for phase_id, phase in phase_list:
            assert phase_id == phase_list.id_from_name(phase.name)

        with pytest.raises(KeyError, match="'d' is not among the phase names "):
            _ = phase_list.id_from_name("d")

    @pytest.mark.parametrize("phase_slice", [slice(0, 3), slice(1, 3), slice(0, 11)])
    def test_get_item_not_indexed(self, phase_slice):
        ids = np.arange(-1, 9)  # [-1, 0, 1, 2, ...]
        pl = PhaseList(ids=ids)  # [-1, 0, 1, 2, ...]
        pl.add_not_indexed()  # [-1, 0, 1, 2, ...]
        assert np.allclose(pl[phase_slice].ids, ids[phase_slice])