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
from scipy.spatial.transform import Rotation as SciPyRotation

from orix.quaternion import Quaternion, Rotation
from orix.vector import Vector3d

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


def test_equality():
    r1 = Rotation.random(5)
    r1_copy = Rotation(r1)
    r2 = Rotation.random(5)
    assert r1 == r1
    assert r1_copy == r1
    # test values not equal
    assert r1 != r2
    # test improper not equal
    r1_copy.improper = ~r1.improper
    assert r1_copy != r1
    # test shape not equal
    r3 = Rotation.random((5, 1))
    assert r3 != r1
    # test not Rotation returns False
    assert r1 != 2
    assert r1 != "test"


@pytest.mark.parametrize(
    "rotation, quaternion, expected",
    [
        ([0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]),
        (
            [0.5, -0.5, -0.5, 0.5],
            [0, np.cos(np.pi / 4), np.sin(np.pi / 4), 0],
            [np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
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
            [np.tan(np.pi / 6), 0, -np.tan(np.pi / 6), np.tan(np.pi / 6)],
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
    ],
    indirect=["rotation"],
)
def test_mul_number(rotation, i, number, expected_i):
    rotation.improper = i
    r = rotation * number
    assert np.allclose(rotation.data, r.data)
    assert np.allclose(r.improper, expected_i)


def test_mul_failing():
    r = Rotation.random()

    with pytest.raises(TypeError):
        _ = r * "cant-mult-by-this"

    for i in [0, -2]:
        with pytest.raises(ValueError, match="Rotations can only be multiplied by"):
            _ = r * i


@pytest.mark.parametrize(
    "rotation, i, expected_i",
    [([0.5, 0.5, 0.5, 0.5], 0, 1), ([0.5, 0.5, 0.5, 0.5], 1, 0)],
    indirect=["rotation"],
)
def test_neg(rotation, i, expected_i):
    rotation.improper = i
    r = -rotation
    assert np.allclose(r.improper, expected_i)


class TestUnique:
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
    def test_unique(self, rotation, improper, expected, improper_expected):
        rotation.improper = improper
        u = rotation.unique()
        assert np.allclose(u.data, expected, atol=1e-6)
        assert np.allclose(u.improper, improper_expected)

    def test_kwargs_unique(self, rotation):
        """return_index and return_inverse edge cases"""
        rotation.unique(return_index=True, return_inverse=True)
        rotation.unique(return_index=True, return_inverse=False)
        rotation.unique(return_index=False, return_inverse=True)

    def test_unique_inverse(self):
        r = Rotation.random(20)
        u, inverse = r.unique(return_inverse=True)
        m = u[inverse] * ~r
        assert np.allclose(m.angle, 0)

    def test_unique_empty(self):
        r = Rotation.empty()
        r2 = r.unique()
        assert r2.size == 0


def test_angle_with_outer():
    shape = (5,)
    r = Rotation.random(shape)
    awo_self = r.angle_with_outer(r)
    assert awo_self.shape == shape + shape
    assert np.allclose(np.diag(awo_self), 0, atol=1e-6)
    r2 = Rotation.random(6)
    dist = r.angle_with_outer(r2)
    assert dist.shape == r.shape + r2.shape
    dist2 = r2.angle_with_outer(r)
    assert dist2.shape == r2.shape + r.shape
    assert np.allclose(dist, dist2.T)

    dist3 = r2.angle_with_outer(r, degrees=True)
    assert np.allclose(dist3, np.rad2deg(dist2))


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
def test_inverse(rotation, improper, expected, improper_expected):
    rotation.improper = improper
    R = ~rotation
    assert np.allclose(R.data, expected, atol=1e-6)
    assert np.allclose(R.improper, improper_expected)

    R2 = rotation.inv()
    assert R == R2


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


def test_outer_lazy_rot():
    r1 = Rotation.random((5, 3))
    r2 = Rotation.random((11, 4))
    r12 = r1.outer(r2)
    r12_lazy = r1.outer(r2, lazy=True, chunk_size=20)
    assert r12.shape == r12_lazy.shape
    assert np.allclose(r12.data, r12_lazy.data)
    assert np.allclose(r12.improper, r12_lazy.improper)
    # different chunk size
    r12_lazy2 = r1.outer(r2, lazy=True, chunk_size=3)
    assert r12.shape == r12_lazy2.shape
    assert np.allclose(r12.data, r12_lazy2.data)
    assert np.allclose(r12.improper, r12_lazy2.improper)


def test_outer_lazy_vec():
    r = Rotation.random((5, 3))
    v = Vector3d.random((6, 4))
    v2 = r.outer(v)
    v2_lazy = r.outer(v, lazy=True, chunk_size=20)
    assert isinstance(v2, Vector3d)
    assert isinstance(v2_lazy, Vector3d)
    assert v2.shape == v2_lazy.shape
    assert np.allclose(v2.data, v2_lazy.data)
    # different chunk size
    v2_lazy2 = r.outer(v, lazy=True, chunk_size=3)
    assert isinstance(v2_lazy2, Vector3d)
    assert v2.shape == v2_lazy.shape
    assert np.allclose(v2.data, v2_lazy2.data)


def test_outer_lazy_progressbar_stdout(capsys):
    r1 = Rotation.random((5, 3))
    r2 = Rotation.random((6, 4))
    _ = r1.outer(r2, lazy=True, progressbar=True)
    out, _ = capsys.readouterr()
    assert "Completed" in out
    _ = r1.outer(r2, lazy=True, progressbar=False)
    out, _ = capsys.readouterr()
    assert not out


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


class TestToFromEuler:
    def test_from_euler(self, eu):
        r = Rotation.from_euler(eu)
        assert isinstance(r, Rotation)


class TestFromToMatrix:
    def test_from_matrix(self):
        r = Rotation.from_matrix(
            [np.identity(3), np.diag([1, -1, -1]), 2 * np.diag([1, -1, -1])]
        )
        assert isinstance(r, Rotation)
        assert np.allclose(r.data, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])

    def test_to_matrix(self):
        r1 = Rotation(
            [
                [0.7071, 0.0, 0.0, 0.7071],
                [0.5, -0.5, -0.5, 0.5],
                [0.0, 0.0, 0.0, 1.0],
                [0.5, 0.5, 0.5, 0.5],
            ]
        )
        om1 = r1.to_matrix()
        r2 = Rotation.from_matrix(om1)
        om2 = r2.to_matrix()
        assert r1 == r2
        assert np.allclose(om1, om2)


class TestFromScipyRotation:
    """These test address the Rotation.from_scipy_rotation()."""

    def test_from_scipy_rotation(self):
        euler = np.deg2rad([15, 32, 41])
        reference_rot = Rotation.from_euler(euler)
        scipy_rot = SciPyRotation.from_euler("ZXZ", euler)  # Bunge convention
        quat = Rotation.from_scipy_rotation(scipy_rot)
        assert np.allclose(reference_rot.angle_with(quat), 0)


class TestFromAlignVectors:
    def test_from_align_vectors(self):
        v1 = Vector3d([[2, -1, 0], [0, 0, 1]])
        v2 = Vector3d([[3, 1, 0], [-1, 3, 0]])
        r12 = Rotation.from_align_vectors(v2, v1)
        assert isinstance(r12, Rotation)
        assert np.allclose((r12 * v1).unit.data, v2.unit.data)
        assert np.allclose((~r12 * v2).unit.data, v1.unit.data)


class TestAngleWith:
    def test_angle_with(self):
        rot1 = Rotation.random(5)
        rot2 = Rotation.random(5)
        ang_rad = rot1.angle_with(rot2)
        ang_deg = rot1.angle_with(rot2, degrees=True)
        assert np.allclose(np.rad2deg(ang_rad), ang_deg)
