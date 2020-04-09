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

import os

import pytest
import numpy as np

from orix import io
from orix.io import load_ang


@pytest.mark.parametrize(
    "angfile_astar, expected_data",
    [
        (
            """4.485496 0.952426 0.791507     0.000     0.000   22.2  0.060  1       6
1.343904 0.276111 0.825890    19.000     0.000   16.3  0.020  1       2
1.343904 0.276111 0.825890    38.000     0.000   18.5  0.030  1       3
1.343904 0.276111 0.825890    57.000     0.000   17.0  0.060  1       6
4.555309 2.895152 3.972020    76.000     0.000   20.5  0.020  1       2
1.361357 0.276111 0.825890    95.000     0.000   16.3  0.010  1       1
4.485496 0.220784 0.810182   114.000     0.000   20.5  0.010  1       1
0.959931 2.369110 4.058938   133.000     0.000   16.5  0.030  1       3
0.959931 2.369110 4.058938   152.000     0.000   16.1  0.030  1       3
4.485496 0.220784 0.810182   171.000     0.000   17.4  0.020  1       2""",
            np.array(
                [
                    [0.77861956, 0.12501022, -0.44104243, -0.42849224],
                    [-0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [-0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [-0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.05331986, -0.95051048, -0.28534763, 0.11074093],
                    [-0.45489991, -0.13271448, -0.03640618, -0.87984517],
                    [0.8752001, 0.02905178, -0.10626836, -0.47104969],
                    [0.3039118, -0.01972273, 0.92612154, -0.22259272],
                    [0.3039118, -0.01972273, 0.92612154, -0.22259272],
                    [0.8752001, 0.02905178, -0.10626836, -0.47104969],
                ]
            ),
        ),
    ],
    indirect=["angfile_astar"],
)
def test_loadang(angfile_astar, expected_data):
    loaded_data = io.loadang(angfile_astar)
    assert np.allclose(loaded_data.data, expected_data)


def test_loadctf():
    """ Crude test of the ctf loader """
    z = np.random.rand(100, 8)
    np.savetxt("temp.ctf", z)
    z_loaded = io.loadctf("temp.ctf")
    os.remove("temp.ctf")


class TestAngReader:
    @pytest.mark.parametrize(
        "angfile_tsl, map_shape, step_sizes, phase_id, n_unknown_columns",
        [
            (
                (
                    (10, 10),  # map_shape
                    (0.1, 0.1),  # step_sizes
                    np.zeros(10 * 10, dtype=int),  # phase_id
                    5,  # n_unknown_columns
                ),
                (10, 10),
                (0.1, 0.1),
                np.zeros(10 * 10, dtype=int),
                5,
            ),
            (
                (
                    (23, 42),  # map_shape
                    (1.5, 1.5),  # step_sizes
                    np.zeros(23 * 42, dtype=int),  # phase_id
                    5,  # n_unknown_columns
                ),
                (23, 42),
                (1.5, 1.5),
                np.zeros(23 * 42, dtype=int),
                5,
            ),
        ],
        indirect=["angfile_tsl"],
    )
    def test_load_ang_tsl(
        self, angfile_tsl, map_shape, step_sizes, phase_id, n_unknown_columns
    ):
        cm = load_ang(angfile_tsl)

        # Fraction of non-indexed points
        non_indexed_fraction = int(np.prod(map_shape) * 0.1)
        assert non_indexed_fraction == np.sum(~cm.is_indexed)

        # Properties
        assert list(cm.prop.keys()) == [
            "iq",
            "ci",
            "unknown1",
            "fit",
            "unknown2",
            "unknown3",
            "unknown4",
            "unknown5",
        ]

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(cm.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(cm.y, np.sort(np.tile(np.arange(ny) * dy, nx)))

        # Ensure that values in columns are within expected ranges or have a certain
        # value
        assert cm.prop["ci"].max() <= 1
        assert cm["indexed"].fit.max() <= 3
        assert all(cm["not_indexed"].ci == -1)
        assert np.allclose(
            cm["not_indexed"].rotations.to_euler()[0],
            np.array([np.pi, 0, np.pi]),
            atol=1e-6,
        )

    def test_load_ang_astar(self):
        pass

    def test_load_ang_emsoft(self):
        pass

    def test_get_header(self):
        pass

    # With different vendor header as input
    def test_vendor_columns(self):
        pass

    # With different vendor header as input
    def test_get_phases_from_header(self):
        pass


class TestEMsoftReader:
    def test_load_emsoft(self):
        pass

    def test_get_properties(self):
        pass
