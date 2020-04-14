# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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

from orix.crystal_map.phase_list import Phase, PhaseList


class TestPhase:
    @pytest.mark.parametrize(
        "name, symmetry, color, color_alias, color_rgb",
        [
            (None, "m-3m", None, "tab:blue", (0.121568, 0.466666, 0.705882)),
            (None, "1", "blue", "b", (0, 0, 1)),
            ("al", "43", "xkcd:salmon", "xkcd:salmon", (1, 0.474509, 0.423529)),
            ("My awes0me phase!", "432", "C1", "tab:orange", (1, 0.498039, 0.054901)),
        ],
    )
    def test_init_phase(self, name, symmetry, color, color_alias, color_rgb):
        p = Phase(name, symmetry, color)

        assert p.name == str(name)

        if symmetry == "43":
            symmetry = "432"
        assert p.symmetry.name == symmetry

        assert p.color == color_alias
        assert np.allclose(p.color_rgb, color_rgb, atol=1e-6)

    @pytest.mark.parametrize("name", [None, "al", 1, np.arange(2)])
    def test_set_phase_name(self, name):
        p = Phase(name=name)
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
        "symmetry, symmetry_name, fails",
        [
            (43, "432", False),
            ("4321", None, True),
            ("m3m", "m-3m", False),
            ("43", "432", False),
        ],
    )
    def test_set_phase_symmetry(self, symmetry, symmetry_name, fails):
        p = Phase()
        if fails:
            with pytest.raises(ValueError, match=f"{symmetry} must be of type"):
                p.symmetry = symmetry
        else:
            p.symmetry = symmetry
            assert p.symmetry.name == symmetry_name

    @pytest.mark.parametrize("name, symmetry", [("al", None), (None, "m-3m")])
    def test_phase_repr_str(self, name, symmetry):
        p = Phase(name=name, symmetry=symmetry, color="C0")
        representation = (
            "<name: "
            + str(name)
            + ". symmetry: "
            + str(symmetry)
            + ". color: tab:blue>"
        )
        assert p.__repr__() == representation
        assert p.__str__() == representation

    def test_phase_deepcopy(self):
        p = Phase("al", "m-3m", "C1")
        p2 = p.deepcopy()

        assert p.__repr__() == "<name: al. symmetry: m-3m. color: tab:orange>"
        p.name = "austenite"
        p.symmetry = 43
        p.color = "C2"

        assert p.__repr__() == "<name: austenite. symmetry: 432. color: tab:green>"
        assert p2.__repr__() == "<name: al. symmetry: m-3m. color: tab:orange>"

    def test_phase_shallowcopy(self):
        p = Phase("al", "m-3m", "C1")
        p2 = p

        p2.name = "austenite"
        p2.symmetry = 43
        p2.color = "C2"

        assert p.__repr__() == p2.__repr__()


class TestPhaseList:
    @pytest.mark.parametrize("empty_input", [(), [], {}])
    def test_init_phaselist_empty_input(self, empty_input):
        pl = PhaseList(empty_input)
        assert pl.__repr__() == "No phases."
        pl["al"] = "m-3m"
        assert pl.__repr__() == (
            "Id  Name  Symmetry     Color\n 0    al      m-3m  tab:blue"
        )

    @pytest.mark.parametrize("phase_collection", ["dict", "list"])
    def test_init_phaselist_from_phases(self, phase_collection):
        p1 = Phase("austenite", 432, None)
        p2 = Phase("ferrite", "432", "C1")
        if phase_collection == "dict":
            phases = {1: p1, 2: p2}
        else:  # phase_collection == "list":
            phases = [p1, p2]

        pl = PhaseList(phases)

        assert pl.names == [p.name for p in [p1, p2]]
        assert pl.symmetries == [p.symmetry for p in [p1, p2]]
        assert pl.colors == [p.color for p in [p1, p2]]
        assert pl.colors_rgb == [p.color_rgb for p in [p1, p2]]

    def test_init_phaselist_from_phase(self):
        p = Phase("austenite", "432", "C2")
        pl = PhaseList(p)

        assert pl.names == [p.name]
        assert pl.symmetries == [p.symmetry]
        assert pl.colors == [p.color]
        assert pl.colors_rgb == [p.color_rgb]

    @pytest.mark.parametrize(
        (
            "names, symmetries, colors, phase_ids, expected_names, "
            "expected_symmetries, expected_colors, expected_phase_ids"
        ),
        [
            (
                ["al", "ni"],
                [43,],
                [None, "C1"],
                [1,],
                ["al", "ni"],
                ["432", None],
                ["tab:blue", "tab:orange"],
                [1, 2],
            ),
            (
                ["al", None],
                [432, "m3m"],
                (1, 0, 0),
                [100,],
                ["al", "None"],
                ["432", "m-3m"],
                ["r", "tab:blue"],
                [100, 101],
            ),
            (
                [None,],
                [None, None],
                ["green", "black"],
                1,
                ["None", "None"],
                [None, None],
                ["g", "k"],
                [1, 2],
            ),
            (
                ["al", "Ni"],
                ["m-3m", 3, None],
                ["C0", None, "C0"],
                None,
                ["al", "Ni", "None"],
                ["m-3m", "3", None],
                ["tab:blue", "tab:orange", "tab:blue"],
                [0, 1, 2],
            ),
        ],
    )
    def test_init_phaselist_from_strings(
        self,
        names,
        symmetries,
        colors,
        phase_ids,
        expected_names,
        expected_symmetries,
        expected_colors,
        expected_phase_ids,
    ):
        pl = PhaseList(
            names=names, symmetries=symmetries, colors=colors, phase_ids=phase_ids,
        )

        actual_symmetry_names = []
        for _, p in pl:
            if p.symmetry is None:
                actual_symmetry_names.append(None)
            else:
                actual_symmetry_names.append(p.symmetry.name)

        assert pl.names == expected_names
        assert actual_symmetry_names == expected_symmetries
        assert pl.colors == expected_colors
        assert pl.phase_ids == expected_phase_ids

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
        "n_names, phase_ids, expected_names, expected_phase_ids",
        [
            (2, [0, 2], ["a", "b"], [0, 2]),
            (3, [1, 100, 2], ["a", "c", "b"], [1, 2, 100]),
            (3, 100, ["a", "b", "c"], [100, 101, 102]),
        ],
    )
    def test_get_phaselist_ids(
        self, n_names, phase_ids, expected_names, expected_phase_ids
    ):
        phase_names_pool = "abc"
        phase_names = [phase_names_pool[i] for i in range(n_names)]

        pl = PhaseList(names=phase_names, phase_ids=phase_ids)

        assert pl.names == expected_names
        assert pl.phase_ids == expected_phase_ids

    @pytest.mark.parametrize(
        "key_getter, name, symmetry, color",
        [
            (0, "a", "m-3m", "r"),
            ("b", "b", "432", "g"),
            (slice(2, None, None), "c", "3", "b"),  # equivalent to pl[2:]
        ],
    )
    def test_get_phase_from_phaselist(
        self, phase_list, key_getter, name, symmetry, color
    ):
        p = phase_list[key_getter]

        assert p.__repr__() == (
            "<name: " + name + ". symmetry: " + symmetry + ". color: " + color + ">"
        )

    @pytest.mark.parametrize(
        "key_getter, names, symmetries, colors",
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
        self, phase_list, key_getter, names, symmetries, colors
    ):
        phases = phase_list[key_getter]

        assert phases.names == names
        assert [p.name for p in phases.symmetries] == symmetries
        assert phases.colors == colors

    @pytest.mark.parametrize("key_getter", ["d", 3, slice(3, None, None)])
    def test_get_from_phaselist_error(self, phase_list, key_getter):
        with pytest.raises(KeyError):
            _ = phase_list[key_getter]

    @pytest.mark.parametrize(
        "key, value, already_there",
        [("d", "m-3m", False), ("d", 432, False), ("c", 432, True),],
    )
    def test_set_phase_in_phaselist(self, phase_list, key, value, already_there):
        if already_there:
            with pytest.raises(ValueError, match=f"{key} is already in the phase "):
                phase_list[key] = value
        else:
            expected_names = phase_list.names + [
                key,
            ]
            expected_symmetry_names = [s.name for s in phase_list.symmetries] + [
                str(value)
            ]

            phase_list[key] = value

            assert phase_list.names == expected_names
            assert [s.name for s in phase_list.symmetries] == expected_symmetry_names

    def test_set_phase_in_empty_phaselist(self):
        pl = PhaseList()

        names = [0, 0]  # Use as names
        symmetries = [432, "m-3m"]
        for n, s in zip(names, symmetries):
            pl[n] = str(s)

        assert pl.phase_ids == [0, 1]
        assert pl.names == [str(n) for n in names]
        assert [s.name for s in pl.symmetries] == [str(s) for s in symmetries]

    @pytest.mark.parametrize(
        "key_del, invalid_phase, error_type, error_msg",
        [
            (0, False, None, None),
            ("a", False, None, None),
            (3, True, KeyError, "3"),
            ("d", True, KeyError, "d is not among the phase names"),
            ([0, 1], True, TypeError, ".* is an invalid phase."),
        ],
    )
    def test_del_phase_in_phaselist(
        self, phase_list, key_del, invalid_phase, error_type, error_msg
    ):
        if invalid_phase:
            with pytest.raises(error_type, match=error_msg):
                del phase_list[key_del]
        else:
            phase_ids = phase_list.phase_ids
            names = phase_list.names

            del phase_list[key_del]

            if isinstance(key_del, int):
                phase_ids.remove(key_del)
                assert phase_list.phase_ids == phase_ids
            elif isinstance(key_del, str):
                names.remove(key_del)
                assert phase_list.names == names

    def test_iterate_phaselist(self):
        names = ["al", "ni", "sigma"]
        symmetries = [3, 432, "m-3m"]
        colors = ["g", "b", "r"]

        pl = PhaseList(names=names, symmetries=symmetries, colors=colors)

        for i, ((phase_id, phase), n, s, c) in enumerate(
            zip(pl, names, symmetries, colors)
        ):
            assert phase_id == i
            assert phase.name == n
            assert phase.symmetry.name == str(s)
            assert phase.color == c

    def test_phaselist_deepcopy(self, phase_list):
        names = phase_list.names
        symmetries = [s.name for s in phase_list.symmetries]
        colors = phase_list.colors

        pl2 = phase_list.deepcopy()
        assert pl2.names == names

        phase_list["d"] = "m-3m"
        phase_list["d"].color = "g"

        assert phase_list.names == names + [
            "d",
        ]
        assert [s.name for s in phase_list.symmetries] == symmetries + [
            "m-3m",
        ]
        assert phase_list.colors == colors + [
            "g",
        ]

        assert pl2.names == names
        assert [s.name for s in pl2.symmetries] == symmetries
        assert pl2.colors == colors

    def test_phaselist_shallowcopy(self, phase_list):
        pl2 = phase_list

        phase_list["d"] = "m-3m"

        assert pl2.names == phase_list.names
        assert [s2.name for s2 in pl2.symmetries] == [
            s.name for s in phase_list.symmetries
        ]
        assert pl2.colors == phase_list.colors

    def test_make_not_indexed(self):
        phase_names = ["a", "b", "c"]
        phase_colors = ["r", "g", "b"]
        pl = PhaseList(names=phase_names, colors=phase_colors, phase_ids=[-1, 0, 1])

        assert pl.names == phase_names
        assert pl.colors == phase_colors

        pl.add_not_indexed()

        phase_names[0] = "not_indexed"
        phase_colors[0] = "w"
        assert pl.names == phase_names
        assert pl.colors == phase_colors
