# -*- coding: utf-8 -*-
# Copyright 2018-2022 the orix developers
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

from math import cos, sin, tan, pi

from diffpy.structure.spacegroups import sg225
import numpy as np
import pytest

from orix.quaternion import Quaternion, Rotation
from orix.vector import AxAngle, Vector3d


rotations = [
    (0.707, 0.0, 0.0, 0.707),
    (0.5, -0.5, -0.5, 0.5),
    (0.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0, 1.0),
    ((0.5, -0.5, -0.5, 0.5), (0.0, 0.0, 0.0, 1.0)),
    Rotation([(2, 4, 6, 8), (-1, -2, -3, -4)]),
    np.array((4, 3, 2, 1)),
]

quaternions = [
    (0.881, 0.665, 0.123, 0.517),
    (0.111, 0.222, 0.333, 0.444),
    ((1, 0, 0.5, 0), (3, 1, -1, -2)),
    [
        [[0.343, 0.343, 0, -0.333], [-7, -8, -9, -10]],
        [[0.00001, -0.0001, 0.001, -0.01], [0, 0, 0, 0]],
    ],
]

vectors = [(1, 0, 0), (1, 1, 0), (0.7, 0.8, 0.9), [[1, 1, 1], [0.4, 0.5, -0.6]]]


@pytest.fixture(params=rotations)
def rotation(request):
    return Rotation(request.param)


rotation_2 = rotation


@pytest.fixture(params=quaternions)
def quaternion(request):
    return Quaternion(request.param)


@pytest.fixture(params=vectors)
def vector(request):
    return Vector3d(request.param)


def test_init(rotation):
    assert np.allclose(rotation.norm, 1)
    assert rotation.improper.shape == rotation.shape
    assert np.count_nonzero(rotation.improper) == 0


def test_slice(rotation):
    r = rotation[0]
    assert np.allclose(r.data, rotation.data[0])
    assert r.improper.shape == r.shape


def test_unit(rotation):
    assert isinstance(rotation.unit, Rotation)
    assert np.allclose(rotation.unit.norm, 1)


@pytest.mark.parametrize(
    "rotation, quaternion, expected",
    [
        ([0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]),
        (
            [0.5, -0.5, -0.5, 0.5],
            [0, cos(pi / 4), sin(pi / 4), 0],
            [cos(pi / 4), 0, sin(pi / 4), 0],
        ),
        (
            [0.794743, 0.50765, -0.33156, 0.0272659],
            [0.545394, 0.358915, 0.569472, 0.499427],
            [0.426441, 0.380997, 0.0280051, 0.819881],
        ),
    ],
    indirect=["rotation", "quaternion"],
)
def test_mul_quaternion(rotation, quaternion, expected):
    r = rotation * quaternion
    assert isinstance(r, Quaternion)
    assert np.allclose(r.data, expected)
    rotation.improper = 1
    ri = rotation * quaternion
    assert np.allclose(r.data, ri.data)


@pytest.mark.parametrize(
    "r1, i1, r2, i2, expected, expected_i",
    [
        ([0.5, 0.5, 0.5, 0.5], 0, [0.5, 0.5, 0.5, 0.5], 0, [-0.5, 0.5, 0.5, 0.5], 0),
        ([0.5, 0.5, 0.5, 0.5], 1, [0.5, 0.5, 0.5, 0.5], 0, [-0.5, 0.5, 0.5, 0.5], 1),
        (
            [0.285883, 0.726947, 0.611896, -0.124108],
            0,
            [-0.247817, -0.574353, 0.594154, 0.505654],
            1,
            [0.0458731, 0.0387992, -0.278082, 0.958677],
            1,
        ),
        (
            [tan(pi / 6), 0, -tan(pi / 6), tan(pi / 6)],
            1,
            [0.5, -0.5, -0.5, 0.5],
            1,
            [-0.288675, -0.288675, -0.866025, 0.288675],
            0,
        ),
    ],
)
def test_mul_rotation(r1, i1, r2, i2, expected, expected_i):
    r1 = Rotation(r1)
    r1.improper = i1
    r2 = Rotation(r2)
    r2.improper = i2
    r = r1 * r2
    assert isinstance(r, Rotation)
    assert np.allclose(r.data, expected)
    assert np.all(r.improper == expected_i)


@pytest.mark.parametrize(
    "rotation, i, vector, expected",
    [
        ([0.5, 0.5, 0.5, 0.5], 0, [1, 1, 0], [0, 1, 1]),
        ([0.5, 0.5, 0.5, 0.5], 1, [1, 1, 0], [0, -1, -1]),
        (
            [-0.172767, -0.346157, 0.664402, -0.63945],
            0,
            [0.237425, -0.813408, 0.531034],
            [0.500697, -0.524764, 0.688422],
        ),
        (
            [-0.172767, -0.346157, 0.664402, -0.63945],
            1,
            [0.237425, -0.813408, 0.531034],
            [-0.500697, 0.524764, -0.688422],
        ),
    ],
    indirect=["rotation", "vector"],
)
def test_mul_vector(rotation, i, vector, expected):
    rotation.improper = i
    v = rotation * vector
    assert isinstance(v, Vector3d)
    assert np.allclose(v.data, expected)


@pytest.mark.parametrize(
    "rotation, i, number, expected_i",
    [
        ([0.5, 0.5, 0.5, 0.5], 0, 1, 0),
        ([0.5, 0.5, 0.5, 0.5], 1, 1, 1),
        ([0.5, 0.5, 0.5, 0.5], 1, -1, 0),
        ([[0, 1, 0, 0], [0, 0, 1, 0]], [0, 1], [-1, 1], [1, 1]),
        ([[0, 1, 0, 0], [0, 0, 1, 0]], [1, 0], [-1, 1], [0, 0]),
        pytest.param([0.5, 0.5, 0.5, 0.5], 1, 2, 0, marks=pytest.mark.xfail),
        pytest.param(
            [0.545394, 0.358915, 0.569472, 0.499427], 0, -2, 0, marks=pytest.mark.xfail
        ),
    ],
    indirect=["rotation"],
)
def test_mul_number(rotation, i, number, expected_i):
    rotation.improper = i
    r = rotation * number
    assert np.allclose(rotation.data, r.data)
    assert np.allclose(r.improper, expected_i)


@pytest.mark.xfail(strict=True, reason=TypeError)
def test_mul_failing(rotation):
    _ = rotation * "cant-mult-by-this"


@pytest.mark.parametrize(
    "rotation, i, expected_i",
    [([0.5, 0.5, 0.5, 0.5], 0, 1), ([0.5, 0.5, 0.5, 0.5], 1, 0)],
    indirect=["rotation"],
)
def test_neg(rotation, i, expected_i):
    rotation.improper = i
    r = -rotation
    assert np.allclose(r.improper, expected_i)


@pytest.fixture()
def e():
    return np.random.rand(10, 3)


class TestToFromEuler:
    """These tests address .to_euler() and .from_euler()."""

    def test_to_from_euler(self, e):
        """Checks that going euler2quat2euler gives no change."""
        r = Rotation.from_euler(e)
        e2 = r.to_euler()
        assert np.allclose(e, e2.data)

    def test_mtex(self, e):
        _ = Rotation.from_euler(e, direction="mtex")

    def test_direction_values(self, e):
        r_mtex = Rotation.from_euler(e, direction="mtex")
        r_c2l = Rotation.from_euler(e, direction="crystal2lab")
        r_l2c = Rotation.from_euler(e, direction="lab2crystal")
        r_default = Rotation.from_euler(e)
        assert np.allclose(r_default.data, r_l2c.data)
        assert np.allclose(r_mtex.data, r_c2l.data)
        assert np.allclose((r_l2c * r_c2l).angle.data, 0)

    def test_direction_kwarg(self, e):
        _ = Rotation.from_euler(e, direction="lab2crystal")

    def test_direction_kwarg_dumb(self, e):
        with pytest.raises(ValueError, match="The chosen direction is not one of "):
            _ = Rotation.from_euler(e, direction="dumb_direction")

    def test_unsupported_conv_to_raises(self, e):
        r = Rotation.from_euler(e)
        with pytest.raises(TypeError, match=r"to_euler\(\) got an unexpected keyword "):
            _ = r.to_euler(convention="bunge")

    def test_unsupported_conv_from_raises(self, e):
        with pytest.raises(ValueError, match="The chosen convention is not one of "):
            _ = Rotation.from_euler(e, convention="unsupported")

    # TODO: remove in 1.0
    def test_conv_from_warns(self, e):
        with pytest.warns(
            np.VisibleDeprecationWarning,
            match=r"Argument `convention` is deprecated and",
        ):
            _ = Rotation.from_euler(e)

    def test_edge_cases_to_euler(self):
        x = np.sqrt(1 / 2)
        q = Rotation(np.asarray([x, 0, 0, x]))
        _ = q.to_euler()
        q = Rotation(np.asarray([0, x, 0, 0]))
        _ = q.to_euler()

    def test_passing_degrees_warns(self):
        with pytest.warns(UserWarning, match="Angles are assumed to be in radians, "):
            r = Rotation.from_euler([90, 0, 0])
            assert np.allclose(r.data, [0.5253, 0, 0, -0.8509], atol=1e-4)


@pytest.mark.parametrize(
    "rotation, improper, expected, improper_expected",
    [
        (
            np.array([[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 1]]),
            [0, 0],
            np.array([[0.5, 0.5, 0.5, 0.5], [0.707106, 0, 0, 0.707106]]),
            [0, 0],
        ),
        (
            np.array([[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 1]]),
            [0, 1],
            np.array([[0.5, 0.5, 0.5, 0.5], [0.707106, 0, 0, 0.707106]]),
            [0, 1],
        ),
        (
            np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
            [0, 0],
            np.array([[0.5, 0.5, 0.5, 0.5]]),
            [0],
        ),
        (
            np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
            [0, 1],
            np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
            [0, 1],
        ),
    ],
    indirect=["rotation"],
)
def test_unique(rotation, improper, expected, improper_expected):
    rotation.improper = improper
    u = rotation.unique()
    assert np.allclose(u.data, expected, atol=1e-6)
    assert np.allclose(u.improper, improper_expected)


def test_kwargs_unique(rotation):
    """return_index and return_inverse edge cases"""
    rotation.unique(return_index=True, return_inverse=True)
    rotation.unique(return_index=True, return_inverse=False)
    rotation.unique(return_index=False, return_inverse=True)


def test_unique_inverse():
    r = Rotation.random((20,))
    u, inverse = r.unique(return_inverse=True)
    m = u[inverse] * ~r
    assert np.allclose(m.angle, 0)


@pytest.mark.parametrize(
    "rotation, improper, expected, improper_expected",
    [
        (
            np.array(
                [
                    [0.231386, 0.270835, 0.779474, 0.515294],
                    [-0.515294, -0.779474, 0.270835, 0.231386],
                ]
            ),
            [0, 1],
            np.array(
                [
                    [0.231386, -0.270835, -0.779474, -0.515294],
                    [-0.515294, 0.779474, -0.270835, -0.231386],
                ]
            ),
            [0, 1],
        ),
    ],
    indirect=["rotation"],
)
def test_inv(rotation, improper, expected, improper_expected):
    rotation.improper = improper
    r = ~rotation
    assert np.allclose(r.data, expected, atol=1e-6)
    assert np.allclose(r.improper, improper_expected)


@pytest.mark.parametrize(
    "rotation, improper, rotation_2, improper_2, expected",
    [
        (
            np.array(
                [
                    [-0.192665, -0.7385, 0.605678, -0.22506],
                    [0.194855, -0.0613995, 0.814759, -0.542614],
                    [-0.440859, -0.61701, -0.305151, 0.576042],
                ]
            ),
            [0, 0, 0],
            np.array(
                [
                    [0.311833, -0.670051, -0.635546, -0.22332],
                    [-0.0608553, -0.380776, -0.662, 0.642699],
                ]
            ),
            [0, 1],
            np.array([[0.1001, 0], [0.2947, 0], [0.3412, 0]]),
        ),
        (
            np.array(
                [
                    [
                        [0.75175, 0.250266, -0.352737, 0.49781],
                        [0.242073, -0.698966, 0.315235, -0.594537],
                        [0.46822, 0.43453, -0.653468, 0.40612],
                        [0.472186, -0.414235, -0.552524, -0.547875],
                        [0.767081, -0.320688, 0.0707849, 0.551122],
                    ],
                    [
                        [-0.507603, -0.63199, -0.441212, 0.385045],
                        [0.775813, 0.122649, -0.616902, -0.0500386],
                        [0.243256, 0.243706, 0.919676, 0.18876],
                        [0.472742, 0.453436, 0.677063, -0.335405],
                        [0.0951788, -0.0223328, 0.924478, -0.368487],
                    ],
                ]
            ),
            np.array([[1, 0, 0, 1, 0], [1, 1, 0, 1, 1]]),
            np.array(
                [
                    [0.733623, -0.289254, -0.51314, -0.338846],
                    [0.654535, 0.491901, 0.544886, -0.180876],
                    [0.529135, 0.166796, -0.329274, 0.764051],
                ]
            ),
            [0, 0, 1],
            np.array(
                [
                    [
                        [0, 0, 0.9360],
                        [0.4195, 0.0939, 0],
                        [0.4155, 0.0907, 0],
                        [0, 0, 0.0559],
                        [0.4324, 0.2832, 0],
                    ],
                    [
                        [0, 0, 0.0655],
                        [0, 0, 0.5959],
                        [0.4279, 0.7461, 0],
                        [0, 0, 0.1534],
                        [0, 0, 0.5393],
                    ],
                ]
            ),
        ),
    ],
    indirect=["rotation", "rotation_2"],
)
def test_dot_outer_rot(rotation, improper, rotation_2, improper_2, expected):
    rotation.improper = improper
    rotation_2.improper = improper_2
    cosines = rotation.dot_outer(rotation_2)
    assert cosines.shape == rotation.shape + rotation_2.shape
    assert np.allclose(cosines.data, expected, atol=1e-4)


@pytest.mark.parametrize(
    "rotation, improper, quaternion, expected",
    [
        (
            np.array(
                [
                    [0.915014, 0.033423, -0.292416, 0.275909],
                    [0.117797, -0.260041, -0.54774, 0.786437],
                    [0.301376, 0.818476, 0.482242, 0.0819321],
                ]
            ),
            [0, 0, 1],
            np.array(
                [
                    [0.15331, -0.0110295, -0.17113, 0.973185],
                    [0.969802, 0.089686, 0.186519, -0.12904],
                ]
            ),
            np.array([[0.4585, 0.8002], [0.8800, 0.1127], [0, 0]]),
        ),
    ],
    indirect=["rotation", "quaternion"],
)
def test_dot_outer_quat(rotation, improper, quaternion, expected):
    rotation.improper = improper
    cosines = rotation.dot_outer(quaternion)
    assert cosines.shape == rotation.shape + quaternion.shape
    assert np.allclose(cosines.data, expected, atol=1e-4)


@pytest.mark.parametrize(
    "rotation, expected",
    [
        ([1, 0, 0, 0], [0, 0, 1]),
        ([-1, 0, 0, 0], [0, 0, -1]),
        ([0, 0.5**0.5, 0.5**0.5, 0], [0.5**0.5, 0.5**0.5, 0]),
        ([[1, 0, 0, 0], [-1, 0, 0, 0]], [[0, 0, 1], [0, 0, -1]]),
    ],
    indirect=["rotation"],
)
def test_axis(rotation, expected):
    ax = rotation.axis
    assert np.allclose(ax.data, expected)


@pytest.mark.parametrize(
    "rotation, improper",
    [
        ([(1, 0, 0, 0), (1, 0, 0, 0)], [0, 1]),
        ([(0.5**0.5, 0, 0, 0.5**0.5)], [1]),
    ],
)
def test_antipodal(rotation, improper):
    rotation = Rotation(rotation)
    rotation.improper = improper
    a = rotation.antipodal
    assert np.allclose(a[0].data, rotation.data)
    assert np.allclose(a[1].data, -rotation.data)
    assert np.allclose(a[0].improper, rotation.improper)
    assert np.allclose(a[1].improper, rotation.improper)


@pytest.mark.parametrize("shape, reference", [((1,), (1, 0, 0, 0))])
def test_random_vonmises(shape, reference):
    r = Rotation.random_vonmises(shape, 1.0, reference)
    assert r.shape == shape
    assert isinstance(r, Rotation)


class TestFromToMatrix:
    def test_to_matrix(self):
        r = Rotation([[1, 0, 0, 0], [3, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])
        om = np.array(
            [np.eye(3), np.eye(3), np.diag([1, -1, -1]), np.diag([1, -1, -1])]
        )
        # Shapes are handled correctly
        assert np.allclose(r.reshape(2, 2).to_matrix(), om.reshape(2, 2, 3, 3))

        r2 = Rotation(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 0.91, 0.92, 0.93],
                [1, 2, 3, 4],
            ]
        )
        om_from_r2 = r2.to_matrix()
        # Inverse equal to transpose
        assert all(np.allclose(np.linalg.inv(i), i.T) for i in om_from_r2)
        # Cross product of any two rows gives the third
        assert all(np.allclose(np.cross(i[:, 0], i[:, 1]), i[:, 2]) for i in om_from_r2)
        # Sum of squares of any column or row equals unity
        assert np.allclose(np.sum(np.square(om_from_r2), axis=1), 1)  # Rows
        assert np.allclose(np.sum(np.square(om_from_r2), axis=2), 1)  # Columns

    def test_from_matrix(self):
        r = Rotation([[1, 0, 0, 0], [3, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])
        om = np.array(
            [np.eye(3), np.eye(3), np.diag([1, -1, -1]), np.diag([1, -1, -1])]
        )
        assert np.allclose(Rotation.from_matrix(om).data, r.data)
        assert np.allclose(
            Rotation.from_matrix(om.reshape((2, 2, 3, 3))).data, r.reshape(2, 2).data
        )

    def test_from_to_matrix(self):
        om = np.array(
            [np.eye(3), np.eye(3), np.diag([1, -1, -1]), np.diag([1, -1, -1])]
        )
        assert np.allclose(Rotation.from_matrix(om).to_matrix(), om)

    def test_from_to_matrix2(self, e):
        r = Rotation.from_euler(e.reshape((5, 2, 3)))
        assert np.allclose(Rotation.from_matrix(r.to_matrix()).data, r.data)

    def test_get_rotation_matrix_from_diffpy(self):
        """Checking that getting rotation matrices from diffpy.structure
        works without issue.
        """
        r = Rotation.from_matrix([i.R for i in sg225.symop_list])
        assert not np.isnan(r.data).any()


class TestFromAxesAngles:
    """These tests address Rotation.from_axes_angles()."""

    def test_from_axes_angles(self, rotations):
        axangle = AxAngle.from_rotation(rotations)
        rotations2 = Rotation.from_neo_euler(axangle)
        rotations3 = Rotation.from_axes_angles(axangle.axis.data, axangle.angle)
        assert np.allclose(rotations2.data, rotations3.data)


def test_to_euler_not_warning():
    rot = Rotation(((0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)))
    with pytest.warns(None) as record:
        _ = rot.to_euler()
    assert len(record) == 0
