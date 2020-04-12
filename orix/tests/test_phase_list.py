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
    def test_init_phaselist(self):
        pass

    def test_get_phaselist_names(self):
        pass

    def test_get_phaselist_colors(self):
        pass

    def test_get_phaselist_colors_rgb(self):
        pass

    def test_get_phaselist_symmetries(self):
        pass

    def test_get_phaselist_size(self):
        pass

    def test_get_phaselist_ids(self):
        pass

    def test_get_phase_from_phaselist(self):
        pass

    def test_set_phase_in_phaselist(self):
        pass

    def test_del_phase_in_phaselist(self):
        pass

    def test_iterate_phaselist(self):
        pass

    def test_phaselist_repr(self):
        pass

    def test_phaselist_deepcopy(self):
        pass

    def test_phaselist_shallowcopy(self):
        pass

    def test_add_not_indexed_to_phaselist(self):
        pass

    def test_sort_phaselist_by_id(self):
        pass
