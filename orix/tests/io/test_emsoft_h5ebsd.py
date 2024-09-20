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
import numpy as np
import pytest

from orix.io import load


class TestEMsoftH5ebsdReader:
    @pytest.mark.parametrize(
        "temp_emsoft_h5ebsd_file, map_shape, step_sizes, n_top_matches, refined",
        [
            (
                (
                    (7, 3),  # map_shape
                    (1.5, 1.5),  # step_sizes
                    np.array(
                        [
                            [6.148271, 0.792205, 1.324879],
                            [6.155951, 0.793078, 1.325229],
                        ]
                    ),  # rotations as rows of Euler angle triplets
                    50,  # n_top_matches
                    True,  # refined
                ),
                (7, 3),
                (1.5, 1.5),
                50,
                True,
            ),
            (
                (
                    (5, 17),
                    (0.5, 0.5),
                    np.array(
                        [
                            [6.148271, 0.792205, 1.324879],
                            [6.155951, 0.793078, 1.325229],
                        ]
                    ),
                    20,
                    False,
                ),
                (5, 17),
                (0.5, 0.5),
                20,
                False,
            ),
        ],
        indirect=["temp_emsoft_h5ebsd_file"],
    )
    def test_load_emsoft(
        self, temp_emsoft_h5ebsd_file, map_shape, step_sizes, n_top_matches, refined
    ):
        xmap = load(temp_emsoft_h5ebsd_file.filename, refined=refined)

        assert xmap.shape == map_shape
        assert (xmap.dy, xmap.dx) == step_sizes
        if refined:
            n_top_matches = 1
        assert xmap.rotations_per_point == n_top_matches

        # Properties
        expected_props = [
            "AvDotProductMap",
            "CI",
            "IQ",
            "ISM",
            "KAM",
            "OSM",
            "TopDotProductList",
            "TopMatchIndices",
        ]
        if refined:
            expected_props += ["RefinedDotProducts"]
        actual_props = list(xmap.prop.keys())
        actual_props.sort()
        expected_props.sort()
        assert actual_props == expected_props

        assert xmap.phases["fe4al13"].structure == Structure(
            title="fe4al13",
            lattice=Lattice(
                a=15.009001, b=8.066, c=12.469, alpha=90, beta=107.72, gamma=90
            ),
        )

        # Ensure Euler angles in degrees are read correctly from file
        if not refined:
            assert np.rad2deg(xmap.rotations.to_euler().min()) >= 150
            assert np.rad2deg(xmap.rotations.to_euler().max()) <= 160

    @pytest.mark.parametrize(
        "temp_emsoft_h5ebsd_file",
        [
            (
                (7, 3),  # map_shape
                (1.5, 1.5),  # step_sizes
                np.array(
                    [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229]]
                ),  # rotations as rows of Euler angle triplets
                21,  # n_top_matches
                False,  # refined
            ),
        ],
        indirect=["temp_emsoft_h5ebsd_file"],
    )
    def test_load_emsoft_best_matches_shape(self, temp_emsoft_h5ebsd_file):
        xmap = load(temp_emsoft_h5ebsd_file.filename, refined=False)
        assert xmap.rotations_per_point == 21
        assert xmap.TopDotProductList.shape == (21, 21)
