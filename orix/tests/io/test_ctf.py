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

from diffpy.structure import Atom, Lattice, Structure
import numpy as np
import pytest

from orix import io
from orix.crystal_map import Phase, PhaseList


class TestCTFReader:
    @pytest.mark.parametrize(
        "ctf_oxford, map_shape, step_sizes, R_example",
        [
            (
                (
                    (5, 3),
                    (0.1, 0.1),
                    np.random.choice([1, 2], 5 * 3),
                    np.array(
                        [[1.59942, 2.37748, 4.53419], [1.59331, 2.37417, 4.53628]]
                    ),
                ),
                (5, 3),
                (0.1, 0.1),
                np.array([[1.59942, 2.37748, 4.53419], [1.59331, 2.37417, 4.53628]]),
            ),
            (
                (
                    (8, 8),
                    (1.0, 1.5),
                    np.random.choice([1, 2], 8 * 8),
                    np.array(
                        [[5.81107, 2.34188, 4.47345], [6.16205, 0.79936, 1.31702]]
                    ),
                ),
                (8, 8),
                (1.0, 1.5),
                np.array([[5.81107, 2.34188, 4.47345], [6.16205, 0.79936, 1.31702]]),
            ),
        ],
        indirect=["ctf_oxford"],
    )
    def test_load_ctf_oxford(
        self,
        ctf_oxford,
        map_shape,
        step_sizes,
        R_example,
    ):
        xmap = io.load(ctf_oxford)

        # Fraction of non-indexed points
        non_indexed_fraction = int(np.prod(map_shape) * 0.1)
        assert non_indexed_fraction == np.sum(~xmap.is_indexed)

        # Properties
        prop_names = ["bands", "error", "MAD", "BC", "BS"]
        assert list(xmap.prop.keys()) == prop_names

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(xmap.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(xmap.y, np.sort(np.tile(np.arange(ny) * dy, nx)))
        assert xmap.scan_unit == "um"

        # Map shape and size
        assert xmap.shape == map_shape
        assert xmap.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert xmap.bands.min() >= 0
        assert xmap.error.min() >= 0
        assert np.allclose(xmap["not_indexed"].bands, 0)
        assert not any(np.isclose(xmap["not_indexed"].error, 0))
        assert np.allclose(xmap["not_indexed"].MAD, 0)
        assert np.allclose(xmap["not_indexed"].BC, 0)
        assert np.allclose(xmap["not_indexed"].BS, 0)

        # Rotations
        R_unique = np.unique(xmap["indexed"].rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(R_unique, axis=0), np.sort(R_example, axis=0), atol=1e-5
        )
        assert np.allclose(xmap["not_indexed"].rotations.to_euler()[0], 0)

        # Phases
        phases = PhaseList(
            names=["Iron fcc", "Iron bcc"],
            space_groups=[225, 229],
            structures=[
                Structure(lattice=Lattice(3.66, 3.66, 3.66, 90, 90, 90)),
                Structure(lattice=Lattice(2.867, 2.867, 2.867, 90, 90, 90)),
            ],
        )

        assert all(np.isin(xmap.phase_id, [-1, 1, 2]))
        assert np.allclose(xmap["not_indexed"].phase_id, -1)
        assert xmap.phases.ids == [-1, 1, 2]
        for (_, phase), (_, phase_test) in zip(xmap["indexed"].phases_in_data, phases):
            assert phase.name == phase_test.name
            assert phase.space_group.number == phase_test.space_group.number
            assert np.allclose(
                phase.structure.lattice.abcABG(), phase_test.structure.lattice.abcABG()
            )

    @pytest.mark.parametrize(
        "ctf_bruker, map_shape, R_example",
        [
            (
                (
                    (5, 3),
                    np.array(
                        [[1.59942, 2.37748, 4.53419], [1.59331, 2.37417, 4.53628]]
                    ),
                ),
                (5, 3),
                np.array([[1.59942, 2.37748, 4.53419], [1.59331, 2.37417, 4.53628]]),
            ),
            (
                (
                    (8, 8),
                    np.array(
                        [[5.81107, 2.34188, 4.47345], [6.16205, 0.79936, 1.31702]]
                    ),
                ),
                (8, 8),
                np.array([[5.81107, 2.34188, 4.47345], [6.16205, 0.79936, 1.31702]]),
            ),
        ],
        indirect=["ctf_bruker"],
    )
    def test_load_ctf_bruker(self, ctf_bruker, map_shape, R_example):
        xmap = io.load(ctf_bruker)

        # Fraction of non-indexed points
        non_indexed_fraction = int(np.prod(map_shape) * 0.1)
        assert non_indexed_fraction == np.sum(~xmap.is_indexed)

        # Properties
        prop_names = ["bands", "error", "MAD", "BC", "BS"]
        assert list(xmap.prop.keys()) == prop_names

        # Coordinates
        ny, nx = map_shape
        dy = dx = 0.001998
        assert np.allclose(xmap.x, np.tile(np.arange(nx) * dx, ny), atol=1e-4)
        assert np.allclose(xmap.y, np.sort(np.tile(np.arange(ny) * dy, nx)), atol=1e-4)
        assert xmap.scan_unit == "um"

        # Map shape and size
        assert xmap.shape == map_shape
        assert xmap.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert xmap.bands.min() >= 0
        assert xmap.error.min() >= 0
        assert np.allclose(xmap["not_indexed"].bands, 0)
        assert not any(np.isclose(xmap["not_indexed"].error, 0))
        assert np.allclose(xmap["not_indexed"].MAD, 0)
        assert np.allclose(xmap["not_indexed"].BC, 0)
        assert np.allclose(xmap["not_indexed"].BS, 0)

        # Rotations
        R_unique = np.unique(xmap["indexed"].rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(R_unique, axis=0), np.sort(R_example, axis=0), atol=1e-5
        )
        assert np.allclose(xmap["not_indexed"].rotations.to_euler()[0], 0)

        # Phases
        assert all(np.isin(xmap.phase_id, [-1, 1]))
        assert np.allclose(xmap["not_indexed"].phase_id, -1)
        assert xmap.phases.ids == [-1, 1]
        phase = xmap.phases[1]
        assert phase.name == "Gold"
        assert phase.space_group.number == 225
