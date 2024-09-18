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

import numpy as np
import pytest

from orix import plot
from orix.vector import Vector3d


class TestFormatVectorLabels:
    @pytest.mark.parametrize(
        "kwargs, desired",
        [
            ({}, ["$111$", "$\\bar{2}01$", "$400$", "$\\bar{4}00$"]),
            (
                dict(brackets=("[", "]"), use_latex=False),
                ["[111]", "[-201]", "[400]", "[-400]"],
            ),
            (
                dict(brackets=("{", "}")),
                [
                    "$\\{111\\}$",
                    "$\\{\\bar{2}01\\}$",
                    "$\\{400\\}$",
                    "$\\{\\bar{4}00\\}$",
                ],
            ),
            (
                dict(brackets=("{", "}"), use_latex=False),
                ["{111}", "{-201}", "{400}", "{-400}"],
            ),
            (
                dict(brackets=("<", ">")),
                [
                    "$\\left<111\\right>$",
                    "$\\left<\\bar{2}01\\right>$",
                    "$\\left<400\\right>$",
                    "$\\left<\\bar{4}00\\right>$",
                ],
            ),
        ],
    )
    def test_format_vector_labels(self, kwargs, desired):
        v = Vector3d([[1, 1, 1], [-2, 0, 1], [4, 0, 0], [-4, 0, 0]])
        v = v.reshape(2, 2)
        labels = plot.format_labels(v.data, **kwargs)
        assert labels.shape == (2, 2)
        assert labels.flatten().tolist() == desired

    def test_format_vector_labels_4(self):
        assert all(
            plot.format_labels([[1, 2, 3, 4], [1.1, 2.2, 3.3, 4.51]])
            == np.array(["$1234$", "$1235$"], dtype="U6")
        )
