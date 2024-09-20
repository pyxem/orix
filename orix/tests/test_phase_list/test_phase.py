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

from diffpy.structure import Atom, Lattice, Structure, loadStructure
import numpy as np
import pytest

from orix.crystal_map import Phase
from orix.crystal_map.phase_list import _new_structure_matrix_from_alignment
from orix.quaternion.symmetry import O, Symmetry


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
        self, name, space_group, desired_sg_str, desired_pg_str, desired_ppg_str
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

    def test_shallow_copy_phase(self):
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

    def test_is_hexagonal(self):
        p1 = Phase(
            point_group="321",
            structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120)),
        )
        p2 = Phase(
            point_group="m-3m",
            structure=Structure(lattice=Lattice(1, 1, 1, 90, 90, 90)),
        )
        assert p1.is_hexagonal
        assert not p2.is_hexagonal

    def test_structure_matrix(self):
        """Structure matrix is updated assuming e1 || a, e3 || c*."""
        trigonal_lattice = Lattice(1.7, 1.7, 1.4, 90, 90, 120)
        phase = Phase(point_group="321", structure=Structure(lattice=trigonal_lattice))
        lattice = phase.structure.lattice

        # Lattice parameters are unchanged
        assert np.allclose(lattice.abcABG(), [1.7, 1.7, 1.4, 90, 90, 120])

        # Structure matrix has changed internally, but not the input
        # `Lattice` instance
        assert not np.allclose(lattice.base, trigonal_lattice.base)

        # The expected structure matrix
        # fmt: off
        assert np.allclose(
            lattice.base,
            [
                [ 1.7,  0,     0  ],
                [-0.85, 1.472, 0  ],
                [ 0,    0,     1.4]
            ],
            atol=1e-3
        )
        # fmt: on

        # Setting the structure also updates the lattice
        phase2 = phase.deepcopy()
        phase2.structure = Structure(lattice=trigonal_lattice)
        assert np.allclose(phase2.structure.lattice.base, lattice.base)

        # Getting new structure matrix without passing enough parameters
        # raises an error
        with pytest.raises(ValueError, match="At least two of x, y, z must be set."):
            _ = _new_structure_matrix_from_alignment(lattice.base, x="a")

    def test_triclinic_structure_matrix(self):
        """Update a triclinic structure matrix."""
        # diffpy.structure aligns e1 || a*, e3 || c* by default
        lat = Lattice(2, 3, 4, 70, 100, 120)
        # fmt: off
        assert np.allclose(
            lat.base,
            [
                [1.732, -0.938, -0.347],
                [0,      2.819,  1.026],
                [0,      0,      4    ]
            ],
            atol=1e-3
        )
        assert np.allclose(
            _new_structure_matrix_from_alignment(lat.base, x="a", z="c*"),
            [
                [ 2,     0,     0    ],
                [-1.5,   2.598, 0    ],
                [-0.695, 1.179, 3.759]
            ],
            atol=1e-3
        )
        assert np.allclose(
            _new_structure_matrix_from_alignment(lat.base, x="b", z="c*"),
            [
                [-1,    -1.732, 0    ],
                [ 3,     0,     0    ],
                [+1.368, 0.012, 3.759]
            ],
            atol=1e-3
        )
        # fmt: on

    def test_lattice_vectors(self):
        """Correct direct and reciprocal lattice vectors."""
        trigonal_lattice = Lattice(1.7, 1.7, 1.4, 90, 90, 120)
        phase = Phase(point_group="321", structure=Structure(lattice=trigonal_lattice))

        a, b, c = phase.a_axis, phase.b_axis, phase.c_axis
        ar, br, cr = phase.ar_axis, phase.br_axis, phase.cr_axis
        # Coordinates in direct and reciprocal crystal reference frames
        assert np.allclose([a.coordinates, ar.coordinates], [1, 0, 0])
        assert np.allclose([b.coordinates, br.coordinates], [0, 1, 0])
        assert np.allclose([c.coordinates, cr.coordinates], [0, 0, 1])
        # Coordinates in cartesian crystal reference frame
        assert np.allclose(a.data, [1.7, 0, 0])
        assert np.allclose(b.data, [-0.85, 1.472, 0], atol=1e-3)
        assert np.allclose(c.data, [0, 0, 1.4])
        assert np.allclose(ar.data, [0.588, 0.340, 0], atol=1e-3)
        assert np.allclose(br.data, [0, 0.679, 0], atol=1e-3)
        assert np.allclose(cr.data, [0, 0, 0.714], atol=1e-3)

    @pytest.mark.parametrize(
        ["lat", "atoms"],
        [
            [
                Lattice(1, 1, 1, 90, 90, 90),
                [
                    Atom("C", [0, 0, 0]),
                    Atom("C", [0.5, 0.5, 0.5]),
                    Atom("C", [0.5, 0, 0]),
                ],
            ],
            [
                Lattice(1, 1, 1, 90, 90, 120),
                [
                    Atom("C", [0, 0, 0]),
                    Atom("C", [0.5, 0, 0]),
                    Atom("C", [0.5, 0.5, 0.5]),
                ],
            ],
            [
                Lattice(1, 2, 3, 90, 90, 60),
                [
                    Atom("C", [0, 0, 0]),
                    Atom("C", [0.1, 0.1, 0.6]),
                    Atom("C", [0.5, 0, 0]),
                ],
            ],
        ],
    )
    def test_atom_positions(self, lat, atoms):
        structure = Structure(atoms, lat)
        phase = Phase(structure=structure)
        # xyz_cartn is independent of basis
        assert np.allclose(phase.structure.xyz_cartn, structure.xyz_cartn)

        # however, Phase should (in many cases) change the basis.
        if np.allclose(structure.lattice.base, phase.structure.lattice.base):
            # In this branch we are in the same basis & all atoms should be the same
            for atom_from_structure, atom_from_phase in zip(structure, phase.structure):
                assert np.allclose(atom_from_structure.xyz, atom_from_phase.xyz)
        else:
            # Here we have differing basis, so xyz must disagree for at least some atoms
            disagreement_found = False

            for atom_from_structure, atom_from_phase in zip(structure, phase.structure):
                if not np.allclose(atom_from_structure.xyz, atom_from_phase.xyz):
                    disagreement_found = True
                    break

            assert disagreement_found

    def test_from_cif(self, cif_file):
        """CIF files parsed correctly with space group and all."""
        phase = Phase.from_cif(cif_file)
        assert phase.space_group.number == 12
        assert phase.point_group.name == "2/m"
        assert len(phase.structure) == 22  # Number of atoms
        lattice = phase.structure.lattice
        assert np.allclose(lattice.abcABG(), [15.5, 4.05, 6.74, 90, 105.3, 90])
        assert np.allclose(
            lattice.base, [[15.5, 0, 0], [0, 4.05, 0], [-1.779, 0, 6.501]], atol=1e-3
        )

    def test_from_cif_same_structure(self, cif_file):
        phase1 = Phase.from_cif(cif_file)
        structure = loadStructure(cif_file)
        phase2 = Phase(structure=structure)
        assert np.allclose(phase1.structure.lattice.base, phase2.structure.lattice.base)
        assert np.allclose(phase1.structure.xyz, phase2.structure.xyz)
