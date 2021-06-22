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

import numpy as np
import pytest

from orix.io import load


class TestBrukerH5ebsdReader:
    @pytest.mark.parametrize(
        "temp_bruker_h5ebsd_file, map_shape, step_sizes, phase_id",
        [
            (
                (
                    (7, 3),  # map_shape
                    (1.5, 1.5),  # step_sizes
                    np.ones(7 * 3, dtype=int),  # phase_id
                    np.array([[1, 2, 3], [4, 5, 6]]),  # rotations
                    False,  # shuffle_order
                ),
                (7, 3),
                (1.5, 1.5),
                np.ones(7 * 3, dtype=int),
            ),
            (
                (
                    (2, 3),
                    (1.4, 1.5),
                    np.concatenate([np.zeros(4), np.ones(2)]).astype(int),
                    np.array([[1, 2, 3], [4, 5, 6]]),
                    True,
                ),
                (2, 3),
                (1.4, 1.5),
                np.concatenate([np.zeros(4), np.ones(2)]),
            ),
        ],
        indirect=["temp_bruker_h5ebsd_file"],
    )
    def test_load_bruker_h5ebsd(
        self, temp_bruker_h5ebsd_file, map_shape, step_sizes, phase_id
    ):
        """Reader works in general case."""
        xmap = load(temp_bruker_h5ebsd_file.filename)

        assert xmap.shape == map_shape
        assert (xmap.dy, xmap.dx) == step_sizes

        # Properties
        expected_props = [
            "DD",
            "MAD",
            "MADPhase",
            "NIndexedBands",
            "PCX",
            "PCY",
            "RadonBandCount",
            "RadonQuality",
            "XBEAM",
            "XSAMPLE",
            "YBEAM",
            "YSAMPLE",
            "ZSAMPLE",
        ]
        actual_props = list(xmap.prop.keys())
        actual_props.sort()
        expected_props.sort()
        assert actual_props == expected_props

        # Phases
        assert xmap.phases.size == np.unique(phase_id).size
        pid = phase_id[-1]
        assert xmap.phases[pid].space_group.number == 225
        assert np.allclose(
            xmap.phases[pid].structure.lattice.abcABG()[3:], [90, 90, 90]
        )
        if 0 in phase_id:
            assert -1 in xmap.phases.ids

        # Rotations
        assert xmap.rotations.to_euler().max() <= 6

        # Coordinate arrays
        map_rows, map_cols = np.indices(map_shape)
        y = map_rows * step_sizes[0]
        x = map_cols * step_sizes[1]
        assert np.allclose(y.ravel(), xmap.y)
        assert np.allclose(x.ravel(), xmap.x)

    def test_different_sem_group_file_location(self, temp_bruker_h5ebsd_file):
        """Reader finds the 'SEM' group HDF5 file location in three
        cases, the final where it is not present at all.
        """
        f = temp_bruker_h5ebsd_file
        _ = load(f.filename)

        f["Scan 1/SEM"] = f["Scan 1/EBSD/SEM"]
        del f["Scan 1/EBSD/SEM"]
        xmap2 = load(temp_bruker_h5ebsd_file.filename)

        del f["Scan 1/SEM"]
        xmap3 = load(temp_bruker_h5ebsd_file.filename)
        assert xmap2.shape == xmap3.shape

    def test_different_sem_dataset_names(self, temp_bruker_h5ebsd_file):
        """Reader finds the 'SEM IY/IX' or 'IY/IX' HDF5 dataset arrays
        both cases.
        """
        f = temp_bruker_h5ebsd_file
        xmap1 = load(f.filename)

        f["Scan 1/EBSD/SEM/SEM IY"] = f["Scan 1/EBSD/SEM/IY"]
        f["Scan 1/EBSD/SEM/SEM IX"] = f["Scan 1/EBSD/SEM/IX"]
        del f["Scan 1/EBSD/SEM/IY"]
        del f["Scan 1/EBSD/SEM/IX"]
        xmap2 = load(temp_bruker_h5ebsd_file.filename)

        assert np.allclose(xmap1.x, xmap2.x)
        assert np.allclose(xmap1.y, xmap2.y)

    def test_non_rectangular_raises(self, temp_bruker_h5ebsd_file):
        """Ensure an explanatory error message is raised the file cannot
        be read.
        """
        f = temp_bruker_h5ebsd_file
        f["Scan 1/EBSD/SEM/IX"][0] = 1001  # Messes up data point order
        with pytest.raises(ValueError, match="Cannot return a crystal map"):
            _ = load(temp_bruker_h5ebsd_file.filename)
