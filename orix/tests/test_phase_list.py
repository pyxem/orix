# -*- coding: utf-8 -*-
# Copyright 2018-2020 the orix developers
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

from orix.crystal_map.phase_list import Phase, PhaseList
from orix.quaternion.symmetry import Symmetry, O


class TestPhase:
    @pytest.mark.parametrize(
        "name, point_group, space_group, color, color_alias, color_rgb, structure",
        [
            (
                None,
                "m-3m",
                None,
                None,
                "tab:blue",
                (0.121568, 0.466666, 0.705882),
                Structure(title="Super", lattice=Lattice(1, 1, 1, 90, 90, 90)),
            ),
            (None, "1", 1, "blue", "b", (0, 0, 1), Structure()),
            (
                "al",
                "43",
                207,
                "xkcd:salmon",
                "xkcd:salmon",
                (1, 0.474509, 0.423529),
                Structure(title="ni", lattice=Lattice(1, 2, 3, 90, 90, 90)),
            ),
            (
                "My awes0me phase!",
                O,
                211,
                "C1",
                "tab:orange",
                (1, 0.498039, 0.054901),
                None,
            ),
        ],
    )
    def test_init_phase(
        self, name, point_group, space_group, color, color_alias, color_rgb, structure
    ):
        p = Phase(
            name=name,
            point_group=point_group,
            space_group=space_group,
            structure=structure,
            color=color,
        )

        if name is None:
            assert p.name == structure.title
        else:
            assert p.name == str(name)

        if space_group is None:
            assert p.space_group is None
        else:
            assert p.space_group.number == space_group

        if point_group == "43":
            point_group = "432"
        if isinstance(point_group, Symmetry):
            point_group = point_group.name
        assert p.point_group.name == point_group

        assert p.color == color_alias
        assert np.allclose(p.color_rgb, color_rgb, atol=1e-6)

        if structure is not None:
            assert p.structure == structure
        else:
            assert p.structure == Structure()

    @pytest.mark.parametrize("name", [None, "al", 1, np.arange(2)])
    def test_set_phase_name(self, name):
        p = Phase(name=name)
        if name is None:
            name = ""
        assert p.name == str(name)

    @pytest.mark.parametrize(
        "color, color_alias, color_rgb, fails",
        [
            ("some-color", None, None, True),
            ("c1", None, None, True),
            ("C1", "tab:orange", (1, 0.498039, 0.054901), False),
        ],
    )
    def test_set_phase_color(self, color, color_alias, color_rgb, fails):
        p = Phase()
        if fails:
            with pytest.raises(ValueError, match="Invalid RGBA argument: "):
                p.color = color
        else:
            p.color = color
            assert p.color == color_alias
            assert np.allclose(p.color_rgb, color_rgb, atol=1e-6)

    @pytest.mark.parametrize(
        "point_group, point_group_name, fails",
        [
            (43, "432", False),
            ("4321", None, True),
            ("m3m", "m-3m", False),
            ("43", "432", False),
        ],
    )
    def test_set_phase_point_group(self, point_group, point_group_name, fails):
        p = Phase()
        if fails:
            with pytest.raises(ValueError, match=f"'{point_group}' must be of type"):
                p.point_group = point_group
        else:
            p.point_group = point_group
            assert p.point_group.name == point_group_name

    @pytest.mark.parametrize(
        "structure", [Structure(), Structure(lattice=Lattice(1, 2, 3, 90, 120, 90))]
    )
    def test_set_structure(self, structure):
        p = Phase()
        p.structure = structure

        assert p.structure == structure

    def test_set_structure_phase_name(self):
        name = "al"
        p = Phase(name=name)
        p.structure = Structure(lattice=Lattice(*([0.405] * 3 + [90] * 3)))
        assert p.name == name
        assert p.structure.title == name

    def test_set_structure_raises(self):
        p = Phase()
        with pytest.raises(ValueError, match=".* must be a diffpy.structure.Structure"):
            p.structure = [1, 2, 3, 90, 90, 90]

    @pytest.mark.parametrize(
        "name, space_group, desired_sg_str, desired_pg_str, desired_ppg_str",
        [
            ("al", None, "None", "None", "None"),
            ("", 207, "P432", "432", "432"),
            ("ni", 225, "Fm-3m", "m-3m", "432"),
        ],
    )
    def test_phase_repr_str(
        self, name, space_group, desired_sg_str, desired_pg_str, desired_ppg_str,
    ):
        p = Phase(name=name, space_group=space_group, color="C0")
        desired = (
            f"<name: {name}. "
            + f"space group: {desired_sg_str}. "
            + f"point group: {desired_pg_str}. "
            + f"proper point group: {desired_ppg_str}. "
            + "color: tab:blue>"
        )
        assert p.__repr__() == desired
        assert p.__str__() == desired

    def test_deepcopy_phase(self):
        p = Phase(name="al", space_group=225, color="C1")
        p2 = p.deepcopy()

        desired_p_repr = (
            "<name: al. space group: Fm-3m. point group: m-3m. proper point group: 432."
            " color: tab:orange>"
        )
        assert p.__repr__() == desired_p_repr

        p.name = "austenite"
        p.space_group = 229
        p.color = "C2"

        new_desired_p_repr = (
            "<name: austenite. space group: Im-3m. point group: m-3m. proper point "
            "group: 432. color: tab:green>"
        )
        assert p.__repr__() == new_desired_p_repr
        assert p2.__repr__() == desired_p_repr

    def test_shallowcopy_phase(self):
        p = Phase(name="al", point_group="m-3m", color="C1")
        p2 = p

        p2.name = "austenite"
        p2.point_group = 43
        p2.color = "C2"

        assert p.__repr__() == p2.__repr__()

    def test_phase_init_non_matching_space_group_point_group(self):
        with pytest.warns(UserWarning, match="Setting space group to 'None', as"):
            _ = Phase(space_group=225, point_group="432")

    @pytest.mark.parametrize(
        "space_group_no, desired_point_group_name",
        [(1, "1"), (50, "mmm"), (100, "4mm"), (150, "32"), (200, "m-3"), (225, "m-3m")],
    )
    def test_point_group_derived_from_space_group(
        self, space_group_no, desired_point_group_name
    ):
        p = Phase(space_group=space_group_no)
        assert p.point_group.name == desired_point_group_name

    def test_set_space_group_raises(self):
        space_group = "outer-space"
        with pytest.raises(ValueError, match=f"'{space_group}' must be of type "):
            p = Phase()
            p.space_group = space_group


class TestPhaseList:
    @pytest.mark.parametrize("empty_input", [(), [], {}])
    def test_init_empty_phaselist(self, empty_input):
        pl = PhaseList(empty_input)
        assert pl.__repr__() == "No phases."
        pl["al"] = "m-3m"
        assert pl.__repr__() == (
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

    @pytest.mark.parametrize(
        "key, value, already_there",
        [("d", "m-3m", False), ("d", 432, False), ("c", 432, True),],
    )
    def test_set_phase_in_phaselist(self, phase_list, key, value, already_there):
        if already_there:
            with pytest.raises(ValueError, match=f"{key} is already in the phase "):
                phase_list[key] = value
        else:
            desired_names = phase_list.names + [key]
            desired_point_group_names = [s.name for s in phase_list.point_groups] + [
                str(value)
            ]

            phase_list[key] = value

            assert phase_list.names == desired_names
            assert [
                s.name for s in phase_list.point_groups
            ] == desired_point_group_names

    def test_set_phase_in_empty_phaselist(self):
        pl = PhaseList()

        names = [0, 0]  # Use as names
        point_groups = [432, "m-3m"]
        for n, s in zip(names, point_groups):
            pl[n] = str(s)

        assert pl.ids == [0, 1]
        assert pl.names == [str(n) for n in names]
        assert [s.name for s in pl.point_groups] == [str(s) for s in point_groups]
        assert pl.structures == [Structure()] * 2

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

        phase_list["d"] = "m-3m"
        phase_list["d"].color = "g"

        assert phase_list.names == names + ["d"]
        assert [s.name for s in phase_list.point_groups] == point_groups + ["m-3m"]
        assert phase_list.colors == colors + ["g"]

        assert pl2.names == names
        assert [s.name for s in pl2.point_groups] == point_groups
        assert pl2.colors == colors

    def test_shallowcopy_phaselist(self, phase_list):
        pl2 = phase_list

        phase_list["d"] = "m-3m"

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
