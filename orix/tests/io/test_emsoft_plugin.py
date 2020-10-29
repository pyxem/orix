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
import pytest
import numpy as np

from orix.io import load


class TestEMsoftPlugin:
    @pytest.mark.parametrize(
        (
            "temp_emsoft_h5ebsd_file, map_shape, step_sizes, example_rot, "
            "n_top_matches, refined"
        ),
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
                np.array(
                    [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
                ),
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
                np.array(
                    [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
                ),
                20,
                False,
            ),
        ],
        indirect=["temp_emsoft_h5ebsd_file"],
    )
    def test_load_emsoft(
        self,
        temp_emsoft_h5ebsd_file,
        map_shape,
        step_sizes,
        example_rot,
        n_top_matches,
        refined,
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
            "CIMap",
            "IQ",
            "IQMap",
            "ISM",
            "ISMap",
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

        assert xmap.phases["austenite"].structure == Structure(
            title="austenite",
            lattice=Lattice(a=3.595, b=3.595, c=3.595, alpha=90, beta=90, gamma=90),
        )

        # Ensure Euler angles in degrees are read correctly from file
        if not refined:
            assert np.rad2deg(xmap.rotations.to_euler().min()) >= 150
            assert np.rad2deg(xmap.rotations.to_euler().max()) <= 160
