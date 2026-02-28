#
# Copyright 2018-2025 the orix developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix. If not, see <http://www.gnu.org/licenses/>.
#

from diffpy.structure import Atom, Lattice, Structure, loadStructure
import numpy as np
import pytest

from orix.crystal_map import Phase
from orix.crystal_map._phase import (
    default_lattice,
    new_structure_matrix_from_alignment,
)
from orix.quaternion.symmetry import O, Symmetry


class TestPhase:
    @pytest.mark.parametrize(
        "name, point_group, space_group, color, color_name, color_rgb, structure",
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
            (None, "1", 1, "blue", "blue", (0, 0, 1), Structure()),
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
        self,
        name,
        point_group,
        space_group,
        color,
        color_name,
        color_rgb,
        structure,
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

        assert p.color == color_name
        assert np.allclose(p.color_rgb, color_rgb, atol=1e-6)

        if structure is not None:
            assert p.structure == structure
        else:
            assert p.structure == Structure()

    def test_copy_constructor_phase(self):
        p1 = Phase(
            "test",
            225,
            "m-3m",
            Structure(
                [Atom("Al", (0, 0, 0))],
                Lattice(10, 10, 10, 90, 90, 90),
            ),
        )
        p2 = Phase(p1)

        assert p1 is not p2
        assert repr(p1) == repr(p2)
        assert p1.structure is not p2.structure
        assert p1.structure[0].element == p2.structure[0].element
        assert tuple(p1.structure[0].xyz) == tuple(p2.structure[0].xyz)
        assert p1.structure[0] is not p2.structure[0]
        assert p1.structure.lattice.abcABG() == p2.structure.lattice.abcABG()
        assert p1.structure.lattice is not p2.structure.lattice

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
            with pytest.raises(
                ValueError, match=f"'{point_group}' could not be interpreted as"
            ):
                p.point_group = point_group
        else:
            p.point_group = point_group
            assert p.point_group.name == point_group_name

    @pytest.mark.parametrize(
        "structure",
        [Structure(), Structure(lattice=Lattice(1, 2, 3, 90, 120, 90))],
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
        with pytest.raises(
            ValueError, match=".* must be a diffpy.structure.Structure"
        ):
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
        with pytest.warns(
            UserWarning, match="Setting space group to 'None', as"
        ):
            _ = Phase(space_group=225, point_group="432")

    @pytest.mark.parametrize(
        "space_group_no, desired_point_group_name",
        [
            (1, "1"),
            (50, "mmm"),
            (100, "4mm"),
            (150, "32"),
            (200, "m-3"),
            (225, "m-3m"),
        ],
    )
    def test_point_group_derived_from_space_group(
        self, space_group_no, desired_point_group_name
    ):
        p = Phase(space_group=space_group_no)
        assert p.point_group.name == desired_point_group_name

    def test_set_space_group_raises(self):
        space_group = "outer-space"
        with pytest.raises(
            ValueError, match=f"'{space_group}' must be of type "
        ):
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
        phase = Phase(
            point_group="321", structure=Structure(lattice=trigonal_lattice)
        )
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
        with pytest.raises(
            ValueError, match="At least two of x, y, z must be set."
        ):
            _ = new_structure_matrix_from_alignment(lattice.base, x="a")

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
            new_structure_matrix_from_alignment(lat.base, x="a", z="c*"),
            [
                [ 2,     0,     0    ],
                [-1.5,   2.598, 0    ],
                [-0.695, 1.179, 3.759]
            ],
            atol=1e-3
        )
        assert np.allclose(
            new_structure_matrix_from_alignment(lat.base, x="b", z="c*"),
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
        phase = Phase(
            point_group="321", structure=Structure(lattice=trigonal_lattice)
        )

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
            for atom_from_structure, atom_from_phase in zip(
                structure, phase.structure
            ):
                assert np.allclose(atom_from_structure.xyz, atom_from_phase.xyz)
        else:
            # Here we have differing basis, so xyz must disagree for at least some atoms
            disagreement_found = False

            for atom_from_structure, atom_from_phase in zip(
                structure, phase.structure
            ):
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
            lattice.base,
            [[15.5, 0, 0], [0, 4.05, 0], [-1.779, 0, 6.501]],
            atol=1e-3,
        )

    def test_from_cif_same_structure(self, cif_file):
        phase1 = Phase.from_cif(cif_file)
        structure = loadStructure(cif_file)
        phase2 = Phase(structure=structure)
        assert np.allclose(
            phase1.structure.lattice.base, phase2.structure.lattice.base
        )
        assert np.allclose(phase1.structure.xyz, phase2.structure.xyz)

    @pytest.mark.parametrize(
        ["lattice", "atoms", "spacegroup", "expected_atom_positions"],
        [
            [
                # P1
                Lattice(1, 1, 1, 90, 90, 90),
                [
                    Atom("C", [0, 0, 0]),
                ],
                1,
                [(0, 0, 0)],
            ],
            [
                # Fd3m
                Lattice(1, 1, 1, 90, 90, 90),
                [
                    Atom("C", [0, 0, 0]),
                ],
                227,
                [
                    (0, 0, 0),
                    (0, 0.5, 0.5),
                    (0.5, 0, 0.5),
                    (0.5, 0.5, 0),
                    (0.25, 0.25, 0.25),
                    (0.25, 0.75, 0.75),
                    (0.75, 0.25, 0.75),
                    (0.75, 0.75, 0.25),
                ],
            ],
            [
                # P63/mmc (graphite)
                Lattice(2, 2, 3, 90, 90, 120),
                [
                    Atom("C", [0, 0, 0.25]),
                    Atom("C", [1 / 3, 2 / 3, 0.75]),
                ],
                194,
                [
                    (0.0, 0.0, 0.25),
                    (0.0, 0.0, 0.75),
                    (0.66666667, 0.33333333, 0.25),
                    (0.33333333, 0.66666667, 0.75),
                ],
            ],
            [
                # https://legacy.materialsproject.org/materials/mp-669458/
                Lattice(
                    8.66993200,
                    14.96934800,
                    22.00995998,
                    90.00000000,
                    91.10362966,
                    90.00000000,
                ),
                [
                    Atom("Cs", (0.00142500, 0.33215200, 0.07744700)),
                    Atom("Cs", (0.00000000, 0.00278300, 0.75000000)),
                    Atom("Bi", (0.01120750, 0.33477050, 0.65575800)),
                    Atom("I", (0.02018650, 0.16620050, 0.58287400)),
                    Atom("I", (0.23071400, 0.07939400, 0.41065700)),
                    Atom("I", (0.23081650, 0.41698250, 0.92495100)),
                    Atom("I", (0.24771900, 0.24833100, 0.24318300)),
                    Atom("I", (0.00000000, 0.49719300, 0.25000000)),
                ],
                15,
                [
                    (0.50000000, 0.50278300, 0.75000000),
                    (0.49857500, 0.83215200, 0.42255300),
                    (0.50000000, 0.49721700, 0.25000000),
                    (0.49857500, 0.16784800, 0.92255300),
                    (0.50142500, 0.16784800, 0.57744700),
                    (0.50142500, 0.83215200, 0.07744700),
                    (0.00000000, 0.00278300, 0.75000000),
                    (0.99857500, 0.33215200, 0.42255300),
                    (0.00000000, 0.99721700, 0.25000000),
                    (0.99857500, 0.66784800, 0.92255300),
                    (0.00142500, 0.66784800, 0.57744700),
                    (0.00142500, 0.33215200, 0.07744700),
                    (0.48879250, 0.83477050, 0.84424200),
                    (0.51120750, 0.16522950, 0.15575800),
                    (0.48879250, 0.16522950, 0.34424200),
                    (0.51120750, 0.83477050, 0.65575800),
                    (0.98879250, 0.33477050, 0.84424200),
                    (0.01120750, 0.66522950, 0.15575800),
                    (0.98879250, 0.66522950, 0.34424200),
                    (0.01120750, 0.33477050, 0.65575800),
                    (0.73081650, 0.08301750, 0.42495100),
                    (0.26918350, 0.91698250, 0.57504900),
                    (0.74771900, 0.25166900, 0.74318300),
                    (0.52018650, 0.33379950, 0.08287400),
                    (0.52018650, 0.66620050, 0.58287400),
                    (0.73081650, 0.91698250, 0.92495100),
                    (0.47981350, 0.66620050, 0.91712600),
                    (0.23071400, 0.92060600, 0.91065700),
                    (0.25228100, 0.74833100, 0.25681700),
                    (0.25228100, 0.25166900, 0.75681700),
                    (0.47981350, 0.33379950, 0.41712600),
                    (0.74771900, 0.74833100, 0.24318300),
                    (0.50000000, 0.00280700, 0.75000000),
                    (0.76928600, 0.92060600, 0.58934300),
                    (0.23071400, 0.07939400, 0.41065700),
                    (0.76928600, 0.07939400, 0.08934300),
                    (0.50000000, 0.99719300, 0.25000000),
                    (0.26918350, 0.08301750, 0.07504900),
                    (0.23081650, 0.58301750, 0.42495100),
                    (0.76918350, 0.41698250, 0.57504900),
                    (0.24771900, 0.75166900, 0.74318300),
                    (0.02018650, 0.83379950, 0.08287400),
                    (0.02018650, 0.16620050, 0.58287400),
                    (0.23081650, 0.41698250, 0.92495100),
                    (0.97981350, 0.16620050, 0.91712600),
                    (0.73071400, 0.42060600, 0.91065700),
                    (0.75228100, 0.24833100, 0.25681700),
                    (0.75228100, 0.75166900, 0.75681700),
                    (0.97981350, 0.83379950, 0.41712600),
                    (0.24771900, 0.24833100, 0.24318300),
                    (0.00000000, 0.50280700, 0.75000000),
                    (0.26928600, 0.42060600, 0.58934300),
                    (0.73071400, 0.57939400, 0.41065700),
                    (0.26928600, 0.57939400, 0.08934300),
                    (0.00000000, 0.49719300, 0.25000000),
                    (0.76918350, 0.58301750, 0.07504900),
                ],
            ],
        ],
    )
    def test_expand_asymmetric_unit(
        self, lattice, atoms, spacegroup, expected_atom_positions
    ):
        s = Structure(lattice=lattice, atoms=atoms)
        phase = Phase(structure=s, space_group=spacegroup)
        base = phase.structure.lattice.base.copy()
        exp = phase.expand_asymmetric_unit()
        assert np.array_equal(base, exp.structure.lattice.base)
        assert len(exp.structure) == len(expected_atom_positions)
        # Check atom positions in ORIGINAL lattice alignment
        # Doing the check in orix's alignment makes independently computing expected sites difficult
        s = exp.structure.copy()
        s.placeInLattice(Lattice(base=phase._diffpy_lattice))
        # Use set to avoid having to ensure the order is the same
        assert set(tuple(xyz.round(8).tolist()) for xyz in s.xyz) == set(
            expected_atom_positions
        )

        # Check that expanding again makes no difference
        exp2 = exp.expand_asymmetric_unit()
        assert np.array_equal(base, exp2.structure.lattice.base)
        assert len(exp2.structure) == len(expected_atom_positions)
        s = exp2.structure.copy()
        s.placeInLattice(Lattice(base=phase._diffpy_lattice))
        assert set(tuple(xyz.round(8).tolist()) for xyz in s.xyz) == set(
            expected_atom_positions
        )

        # Check that original phase was preserved
        assert len(phase.structure) == len(atoms)
        assert np.array_equal(phase.structure.lattice.base, base)

    @pytest.mark.parametrize(
        ["cif_file_content", "expected_atom_count"],
        [
            (
                # P1
                """
                # generated using pymatgen
                data_Si
                _symmetry_space_group_name_H-M   P1
                _cell_length_a   5.46872800
                _cell_length_b   5.46872800
                _cell_length_c   5.46872800
                _cell_angle_alpha   90.00000000
                _cell_angle_beta   90.00000000
                _cell_angle_gamma   90.00000000
                _symmetry_Int_Tables_number   1
                _chemical_formula_structural   Si
                _chemical_formula_sum   Si1
                _cell_volume   163.55317139
                loop_
                _symmetry_equiv_pos_site_id
                _symmetry_equiv_pos_as_xyz
                1  'x, y, z'
                loop_
                _atom_site_type_symbol
                _atom_site_label
                _atom_site_fract_x
                _atom_site_fract_y
                _atom_site_fract_z
                _atom_site_occupancy
                Si  Si0  0.00000000  0.00000000  0.50000000  1
                """.lstrip(),
                1,
            ),
            (
                # Graphite, two atoms in asymmetric -> 4 in expanded
                """
                # generated using pymatgen
                data_C
                _symmetry_space_group_name_H-M   P6_3/mmc
                _cell_length_a   2.46772414
                _cell_length_b   2.46772414
                _cell_length_c   8.68503800
                _cell_angle_alpha   90.00000000
                _cell_angle_beta   90.00000000
                _cell_angle_gamma   120.00000000
                _symmetry_Int_Tables_number   194
                _chemical_formula_structural   C
                _chemical_formula_sum   C4
                _cell_volume   45.80317400
                _cell_formula_units_Z   4
                loop_
                _symmetry_equiv_pos_site_id
                _symmetry_equiv_pos_as_xyz
                1  'x, y, z'
                2  '-x, -y, -z'
                3  'x-y, x, z+1/2'
                4  '-x+y, -x, -z+1/2'
                5  '-y, x-y, z'
                6  'y, -x+y, -z'
                7  '-x, -y, z+1/2'
                8  'x, y, -z+1/2'
                9  '-x+y, -x, z'
                10  'x-y, x, -z'
                11  'y, -x+y, z+1/2'
                12  '-y, x-y, -z+1/2'
                13  '-y, -x, -z+1/2'
                14  'y, x, z+1/2'
                15  '-x, -x+y, -z'
                16  'x, x-y, z'
                17  '-x+y, y, -z+1/2'
                18  'x-y, -y, z+1/2'
                19  'y, x, -z'
                20  '-y, -x, z'
                21  'x, x-y, -z+1/2'
                22  '-x, -x+y, z+1/2'
                23  'x-y, -y, -z'
                24  '-x+y, y, z'
                loop_
                _atom_site_type_symbol
                _atom_site_label
                _atom_site_symmetry_multiplicity
                _atom_site_fract_x
                _atom_site_fract_y
                _atom_site_fract_z
                _atom_site_occupancy
                C  C0  2  0.00000000  0.00000000  0.25000000  1
                C  C1  2  0.33333333  0.66666667  0.75000000  1
                """.lstrip(),
                4,
            ),
            (
                # Si Fd3m. 1 asymmetric -> 8 expanded
                """
                # generated using pymatgen
                data_Si
                _symmetry_space_group_name_H-M   Fd-3m
                _cell_length_a   5.46872800
                _cell_length_b   5.46872800
                _cell_length_c   5.46872800
                _cell_angle_alpha   90.00000000
                _cell_angle_beta   90.00000000
                _cell_angle_gamma   90.00000000
                _symmetry_Int_Tables_number   227
                _chemical_formula_structural   Si
                _chemical_formula_sum   Si8
                _cell_volume   163.55317139
                _cell_formula_units_Z   8
                loop_
                _symmetry_equiv_pos_site_id
                _symmetry_equiv_pos_as_xyz
                1  'x, y, z'
                2  '-y+1/4, x+3/4, z+3/4'
                3  '-x, -y, z'
                4  'y+1/4, -x+3/4, z+3/4'
                5  'x, -y, -z'
                6  '-y+1/4, -x+3/4, -z+3/4'
                7  '-x, y, -z'
                8  'y+1/4, x+3/4, -z+3/4'
                9  'z, x, y'
                10  'z+1/4, -y+3/4, x+3/4'
                11  'z, -x, -y'
                12  'z+1/4, y+3/4, -x+3/4'
                13  '-z, x, -y'
                14  '-z+1/4, -y+3/4, -x+3/4'
                15  '-z, -x, y'
                16  '-z+1/4, y+3/4, x+3/4'
                17  'y, z, x'
                18  'x+1/4, z+3/4, -y+3/4'
                19  '-y, z, -x'
                20  '-x+1/4, z+3/4, y+3/4'
                21  '-y, -z, x'
                22  '-x+1/4, -z+3/4, -y+3/4'
                23  'y, -z, -x'
                24  'x+1/4, -z+3/4, y+3/4'
                25  '-x+1/4, -y+3/4, -z+3/4'
                26  'y, -x, -z'
                27  'x+1/4, y+3/4, -z+3/4'
                28  '-y, x, -z'
                29  '-x+1/4, y+3/4, z+3/4'
                30  'y, x, z'
                31  'x+1/4, -y+3/4, z+3/4'
                32  '-y, -x, z'
                33  '-z+1/4, -x+3/4, -y+3/4'
                34  '-z, y, -x'
                35  '-z+1/4, x+3/4, y+3/4'
                36  '-z, -y, x'
                37  'z+1/4, -x+3/4, y+3/4'
                38  'z, y, x'
                39  'z+1/4, x+3/4, -y+3/4'
                40  'z, -y, -x'
                41  '-y+1/4, -z+3/4, -x+3/4'
                42  '-x, -z, y'
                43  'y+1/4, -z+3/4, x+3/4'
                44  'x, -z, -y'
                45  'y+1/4, z+3/4, -x+3/4'
                46  'x, z, y'
                47  '-y+1/4, z+3/4, x+3/4'
                48  '-x, z, -y'
                49  'x+1/2, y+1/2, z'
                50  '-y+3/4, x+1/4, z+3/4'
                51  '-x+1/2, -y+1/2, z'
                52  'y+3/4, -x+1/4, z+3/4'
                53  'x+1/2, -y+1/2, -z'
                54  '-y+3/4, -x+1/4, -z+3/4'
                55  '-x+1/2, y+1/2, -z'
                56  'y+3/4, x+1/4, -z+3/4'
                57  'z+1/2, x+1/2, y'
                58  'z+3/4, -y+1/4, x+3/4'
                59  'z+1/2, -x+1/2, -y'
                60  'z+3/4, y+1/4, -x+3/4'
                61  '-z+1/2, x+1/2, -y'
                62  '-z+3/4, -y+1/4, -x+3/4'
                63  '-z+1/2, -x+1/2, y'
                64  '-z+3/4, y+1/4, x+3/4'
                65  'y+1/2, z+1/2, x'
                66  'x+3/4, z+1/4, -y+3/4'
                67  '-y+1/2, z+1/2, -x'
                68  '-x+3/4, z+1/4, y+3/4'
                69  '-y+1/2, -z+1/2, x'
                70  '-x+3/4, -z+1/4, -y+3/4'
                71  'y+1/2, -z+1/2, -x'
                72  'x+3/4, -z+1/4, y+3/4'
                73  '-x+3/4, -y+1/4, -z+3/4'
                74  'y+1/2, -x+1/2, -z'
                75  'x+3/4, y+1/4, -z+3/4'
                76  '-y+1/2, x+1/2, -z'
                77  '-x+3/4, y+1/4, z+3/4'
                78  'y+1/2, x+1/2, z'
                79  'x+3/4, -y+1/4, z+3/4'
                80  '-y+1/2, -x+1/2, z'
                81  '-z+3/4, -x+1/4, -y+3/4'
                82  '-z+1/2, y+1/2, -x'
                83  '-z+3/4, x+1/4, y+3/4'
                84  '-z+1/2, -y+1/2, x'
                85  'z+3/4, -x+1/4, y+3/4'
                86  'z+1/2, y+1/2, x'
                87  'z+3/4, x+1/4, -y+3/4'
                88  'z+1/2, -y+1/2, -x'
                89  '-y+3/4, -z+1/4, -x+3/4'
                90  '-x+1/2, -z+1/2, y'
                91  'y+3/4, -z+1/4, x+3/4'
                92  'x+1/2, -z+1/2, -y'
                93  'y+3/4, z+1/4, -x+3/4'
                94  'x+1/2, z+1/2, y'
                95  '-y+3/4, z+1/4, x+3/4'
                96  '-x+1/2, z+1/2, -y'
                97  'x+1/2, y, z+1/2'
                98  '-y+3/4, x+3/4, z+1/4'
                99  '-x+1/2, -y, z+1/2'
                100  'y+3/4, -x+3/4, z+1/4'
                101  'x+1/2, -y, -z+1/2'
                102  '-y+3/4, -x+3/4, -z+1/4'
                103  '-x+1/2, y, -z+1/2'
                104  'y+3/4, x+3/4, -z+1/4'
                105  'z+1/2, x, y+1/2'
                106  'z+3/4, -y+3/4, x+1/4'
                107  'z+1/2, -x, -y+1/2'
                108  'z+3/4, y+3/4, -x+1/4'
                109  '-z+1/2, x, -y+1/2'
                110  '-z+3/4, -y+3/4, -x+1/4'
                111  '-z+1/2, -x, y+1/2'
                112  '-z+3/4, y+3/4, x+1/4'
                113  'y+1/2, z, x+1/2'
                114  'x+3/4, z+3/4, -y+1/4'
                115  '-y+1/2, z, -x+1/2'
                116  '-x+3/4, z+3/4, y+1/4'
                117  '-y+1/2, -z, x+1/2'
                118  '-x+3/4, -z+3/4, -y+1/4'
                119  'y+1/2, -z, -x+1/2'
                120  'x+3/4, -z+3/4, y+1/4'
                121  '-x+3/4, -y+3/4, -z+1/4'
                122  'y+1/2, -x, -z+1/2'
                123  'x+3/4, y+3/4, -z+1/4'
                124  '-y+1/2, x, -z+1/2'
                125  '-x+3/4, y+3/4, z+1/4'
                126  'y+1/2, x, z+1/2'
                127  'x+3/4, -y+3/4, z+1/4'
                128  '-y+1/2, -x, z+1/2'
                129  '-z+3/4, -x+3/4, -y+1/4'
                130  '-z+1/2, y, -x+1/2'
                131  '-z+3/4, x+3/4, y+1/4'
                132  '-z+1/2, -y, x+1/2'
                133  'z+3/4, -x+3/4, y+1/4'
                134  'z+1/2, y, x+1/2'
                135  'z+3/4, x+3/4, -y+1/4'
                136  'z+1/2, -y, -x+1/2'
                137  '-y+3/4, -z+3/4, -x+1/4'
                138  '-x+1/2, -z, y+1/2'
                139  'y+3/4, -z+3/4, x+1/4'
                140  'x+1/2, -z, -y+1/2'
                141  'y+3/4, z+3/4, -x+1/4'
                142  'x+1/2, z, y+1/2'
                143  '-y+3/4, z+3/4, x+1/4'
                144  '-x+1/2, z, -y+1/2'
                145  'x, y+1/2, z+1/2'
                146  '-y+1/4, x+1/4, z+1/4'
                147  '-x, -y+1/2, z+1/2'
                148  'y+1/4, -x+1/4, z+1/4'
                149  'x, -y+1/2, -z+1/2'
                150  '-y+1/4, -x+1/4, -z+1/4'
                151  '-x, y+1/2, -z+1/2'
                152  'y+1/4, x+1/4, -z+1/4'
                153  'z, x+1/2, y+1/2'
                154  'z+1/4, -y+1/4, x+1/4'
                155  'z, -x+1/2, -y+1/2'
                156  'z+1/4, y+1/4, -x+1/4'
                157  '-z, x+1/2, -y+1/2'
                158  '-z+1/4, -y+1/4, -x+1/4'
                159  '-z, -x+1/2, y+1/2'
                160  '-z+1/4, y+1/4, x+1/4'
                161  'y, z+1/2, x+1/2'
                162  'x+1/4, z+1/4, -y+1/4'
                163  '-y, z+1/2, -x+1/2'
                164  '-x+1/4, z+1/4, y+1/4'
                165  '-y, -z+1/2, x+1/2'
                166  '-x+1/4, -z+1/4, -y+1/4'
                167  'y, -z+1/2, -x+1/2'
                168  'x+1/4, -z+1/4, y+1/4'
                169  '-x+1/4, -y+1/4, -z+1/4'
                170  'y, -x+1/2, -z+1/2'
                171  'x+1/4, y+1/4, -z+1/4'
                172  '-y, x+1/2, -z+1/2'
                173  '-x+1/4, y+1/4, z+1/4'
                174  'y, x+1/2, z+1/2'
                175  'x+1/4, -y+1/4, z+1/4'
                176  '-y, -x+1/2, z+1/2'
                177  '-z+1/4, -x+1/4, -y+1/4'
                178  '-z, y+1/2, -x+1/2'
                179  '-z+1/4, x+1/4, y+1/4'
                180  '-z, -y+1/2, x+1/2'
                181  'z+1/4, -x+1/4, y+1/4'
                182  'z, y+1/2, x+1/2'
                183  'z+1/4, x+1/4, -y+1/4'
                184  'z, -y+1/2, -x+1/2'
                185  '-y+1/4, -z+1/4, -x+1/4'
                186  '-x, -z+1/2, y+1/2'
                187  'y+1/4, -z+1/4, x+1/4'
                188  'x, -z+1/2, -y+1/2'
                189  'y+1/4, z+1/4, -x+1/4'
                190  'x, z+1/2, y+1/2'
                191  '-y+1/4, z+1/4, x+1/4'
                192  '-x, z+1/2, -y+1/2'
                loop_
                _atom_site_type_symbol
                _atom_site_label
                _atom_site_symmetry_multiplicity
                _atom_site_fract_x
                _atom_site_fract_y
                _atom_site_fract_z
                _atom_site_occupancy
                Si  Si0  8  0.00000000  0.00000000  0.50000000  1
                """.lstrip(),
                8,
            ),
            (
                # https://legacy.materialsproject.org/materials/mp-669458/
                """
                # generated using pymatgen
                data_Cs3Bi2I9
                _symmetry_space_group_name_H-M   C2/c
                _cell_length_a   8.66993200
                _cell_length_b   14.96934800
                _cell_length_c   22.00995998
                _cell_angle_alpha   90.00000000
                _cell_angle_beta   91.10362966
                _cell_angle_gamma   90.00000000
                _symmetry_Int_Tables_number   15
                _chemical_formula_structural   Cs3Bi2I9
                _chemical_formula_sum   'Cs12 Bi8 I36'
                _cell_volume   2855.99377941
                _cell_formula_units_Z   4
                loop_
                _symmetry_equiv_pos_site_id
                _symmetry_equiv_pos_as_xyz
                1  'x, y, z'
                2  '-x, -y, -z'
                3  '-x, y, -z+1/2'
                4  'x, -y, z+1/2'
                5  'x+1/2, y+1/2, z'
                6  '-x+1/2, -y+1/2, -z'
                7  '-x+1/2, y+1/2, -z+1/2'
                8  'x+1/2, -y+1/2, z+1/2'
                loop_
                _atom_site_type_symbol
                _atom_site_label
                _atom_site_symmetry_multiplicity
                _atom_site_fract_x
                _atom_site_fract_y
                _atom_site_fract_z
                _atom_site_occupancy
                Cs  Cs0  8  0.00142500  0.33215200  0.07744700  1
                Cs  Cs1  4  0.00000000  0.00278300  0.75000000  1
                Bi  Bi2  8  0.01120750  0.33477050  0.65575800  1
                I  I3  8  0.02018650  0.16620050  0.58287400  1
                I  I4  8  0.23071400  0.07939400  0.41065700  1
                I  I5  8  0.23081650  0.41698250  0.92495100  1
                I  I6  8  0.24771900  0.24833100  0.24318300  1
                I  I7  4  0.00000000  0.49719300  0.25000000  1
                """.lstrip(),
                # Sum the multiplicities from the cif
                8 + 4 + 8 + 8 + 8 + 8 + 8 + 4,
            ),
        ],
    )
    def test_expand_asymmetric_unit_from_cif(
        self, cif_file_content, expected_atom_count, tmp_path
    ):
        filepath = tmp_path / "tmp.cif"
        with open(filepath, "w") as file:
            file.write(cif_file_content)
        phase = Phase.from_cif(filepath)
        # Asymmetric unit is automatically expanded when read from cif
        assert len(phase.structure) == expected_atom_count
        # Expand just in case
        exp = phase.expand_asymmetric_unit()
        assert len(exp.structure) == expected_atom_count

    def test_expand_asymmetric_unit_raise_if_no_point_group(self):
        phase = Phase()
        with pytest.raises(ValueError, match="Space group must be set"):
            phase.expand_asymmetric_unit()

    def test_default_lattice(self):
        for S in ["1", "2", "222", "422", "432"]:
            phase = Phase(point_group=S)
            lattice_parameters = phase.structure.lattice.abcABG()
            assert np.allclose([1, 1, 1, 90, 90, 90], lattice_parameters)

        for S in ["3", "622"]:
            phase = Phase(point_group=S)
            lattice_parameters = phase.structure.lattice.abcABG()
            assert np.allclose([1, 1, 1, 90, 90, 120], lattice_parameters)

    def test_default_lattice_raises(self):
        with pytest.raises(
            ValueError, match="Unknown crystal system 'rhombohedral'"
        ):
            default_lattice("rhombohedral")
