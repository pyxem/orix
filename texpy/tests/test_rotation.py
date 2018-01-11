from math import cos, sin, tan, pi
import numpy as np
import pytest
import itertools

from texpy.quaternion.quaternion import Quaternion
from texpy.quaternion.rotation import Rotation
from texpy.vector.vector3d import Vector3d

rotations = [
    (0.707, 0., 0., 0.707),
    (0.5, -0.5, -0.5, 0.5),
    (0., 0., 0., 1.),
    (1., 1., 1., 1.),
    (
        (0.5, -0.5, -0.5, 0.5),
        (0., 0., 0., 1.),
    ),
    Rotation([
        (2, 4, 6, 8),
        (-1, -2, -3, -4)
    ]),
    np.array((4, 3, 2, 1))
]

quaternions = [
    (0.881, 0.665, 0.123, 0.517),
    (0.111, 0.222, 0.333, 0.444),
    (
        (1, 0, 0.5, 0),
        (3, 1, -1, -2),
    ),
    [
        [[0.343, 0.343, 0, -0.333], [-7, -8, -9, -10],],
        [[0.00001, -0.0001, 0.001, -0.01], [0, 0, 0, 0]]
    ]
]

vectors = [
    (1, 0, 0),
    (1, 1, 0),
    (0.7, 0.8, 0.9),
    [
        [1, 1, 1],
        [0.4, 0.5, -0.6],
    ]
]


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
    assert np.all(rotation.improper == False)


def test_slice(rotation):
    r = rotation[0]
    assert np.allclose(r.data, rotation.data[0])
    assert r.improper.shape == r.shape


def test_unit(rotation):
    assert isinstance(rotation.unit, Rotation)
    assert np.allclose(rotation.unit.norm, 1)


@pytest.mark.parametrize('rotation, quaternion, expected', [
    ([0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]),
    ([0.5, -0.5, -0.5, 0.5], [0, cos(pi/4), sin(pi/4), 0], [cos(pi/4), 0, sin(pi/4), 0]),
    ([0.794743, 0.50765,  -0.33156, 0.0272659], [0.545394, 0.358915, 0.569472, 0.499427], [0.426441, 0.380997, 0.0280051, 0.819881]),
], indirect=['rotation', 'quaternion'])
def test_mul_quaternion(rotation, quaternion, expected):
    r = rotation * quaternion
    assert isinstance(r, Quaternion)
    assert np.allclose(r.data, expected)
    rotation.improper = 1
    ri = rotation * quaternion
    assert np.allclose(r.data, ri.data)


@pytest.mark.parametrize('r1, i1, r2, i2, expected, expected_i', [
    ([0.5, 0.5, 0.5, 0.5], 0, [0.5, 0.5, 0.5, 0.5], 0, [-0.5, 0.5, 0.5, 0.5], 0),
    ([0.5, 0.5, 0.5, 0.5], 1, [0.5, 0.5, 0.5, 0.5], 0, [-0.5, 0.5, 0.5, 0.5], 1),
    ([0.285883, 0.726947, 0.611896, -0.124108], 0, [-0.247817, -0.574353, 0.594154, 0.505654], 1, [0.0458731, 0.0387992, -0.278082,  0.958677], 1),
    ([tan(pi/6), 0, -tan(pi/6), tan(pi/6)], 1, [0.5, -0.5, -0.5, 0.5], 1, [-0.288675, -0.288675, -0.866025, 0.288675], 0)
])
def test_mul_rotation(r1, i1, r2, i2, expected, expected_i):
    r1 = Rotation(r1)
    r1.improper = i1
    r2 = Rotation(r2)
    r2.improper = i2
    r = r1 * r2
    assert isinstance(r, Rotation)
    assert np.allclose(r.data, expected)
    assert np.all(r.improper == expected_i)


@pytest.mark.parametrize('rotation, i, vector, expected', [
    ([0.5, 0.5, 0.5, 0.5], 0, [1, 1, 0], [0, 1, 1]),
    ([0.5, 0.5, 0.5, 0.5], 1, [1, 1, 0], [0, -1, -1]),
    ([-0.172767, -0.346157, 0.664402, -0.63945], 0, [0.237425, -0.813408, 0.531034], [0.500697, -0.524764, 0.688422]),
    ([-0.172767, -0.346157, 0.664402, -0.63945], 1, [0.237425, -0.813408, 0.531034], [-0.500697, 0.524764, -0.688422])
], indirect=['rotation', 'vector'])
def test_mul_vector(rotation, i, vector, expected):
    rotation.improper = i
    v = rotation * vector
    assert isinstance(v, Vector3d)
    assert np.allclose(v.data, expected)


@pytest.mark.parametrize('rotation, i, number, expected_i', [
    ([0.5, 0.5, 0.5, 0.5], 0, 1, 0),
    ([0.5, 0.5, 0.5, 0.5], 1, 1, 1),
    ([0.5, 0.5, 0.5, 0.5], 1, -1, 0),
    ([[0, 1, 0, 0], [0, 0, 1, 0]], [0, 1], [-1, 1], [1, 1]),
    ([[0, 1, 0, 0], [0, 0, 1, 0]], [1, 0], [-1, 1], [0, 0]),
    pytest.param([0.5, 0.5, 0.5, 0.5], 1, 2, 0, marks=pytest.mark.xfail),
    pytest.param([0.545394, 0.358915, 0.569472, 0.499427], 0, -2, 0, marks=pytest.mark.xfail),
], indirect=['rotation'])
def test_mul_number(rotation, i, number, expected_i):
    rotation.improper = i
    r = rotation * number
    assert np.allclose(rotation.data, r.data)
    assert np.allclose(r.improper, expected_i)


@pytest.mark.parametrize('rotation, i, expected_i', [
    ([0.5, 0.5, 0.5, 0.5], 0, 1),
    ([0.5, 0.5, 0.5, 0.5], 1, 0),
], indirect=['rotation'])
def test_neg(rotation, i, expected_i):
    rotation.improper = i
    r = -rotation
    assert np.allclose(r.improper, expected_i)

min_axes = [
    (0, 0, 1),
    (-0.57735, -0.57735,  0.57735),
    (0, 0, 1),
    (0.57735, 0.57735, 0.57735),
    np.array((
        (-0.57735, -0.57735,  0.57735),
        (0, 0, 1),
    )),
    np.array([
        [0.371391, 0.557086, 0.742781],
        [0.371391, 0.557086, 0.742781],
    ]),
    (0.801784, 0.534522, 0.267261),
]


@pytest.mark.parametrize('rotation, min_axes', zip(rotations, min_axes))
def test_min_axes(rotation, min_axes):
    ma = Rotation(rotation).min_axes()
    assert np.allclose(ma.data, min_axes, atol=1e-3)


@pytest.mark.parametrize('rotation, euler, convention', [
    ([0.408454, 0.500391, -0.761426, 0.054816], [310.956, 131.324, 64.3317], 'bunge'),
    pytest.param([-0.155403, -0.493385, -0.0692231, -0.853012], [87.6616, 59.7643, 71.6884], 'xyz', marks=pytest.mark.xfail),
    ([
        [ 0.636895, -0.258037,  0.616387,  -0.38451],
        [-0.274124, -0.484693,  0.366742,  0.745271],
        [ 0.518714, -0.101143, -0.065675,    0.8464],
        [ 0.742955, -0.475664, -0.373451, -0.286873],
    ],
    [
        [81.5951, 83.8595, 216.164],
        [253.082,  74.862, 327.307],
        [271.495, 13.8528, 205.501],
        [197.023, 74.4216, 120.751],
    ], 'zxz'),
], indirect=['rotation'])
def test_to_euler(rotation, euler, convention):
    e = np.degrees(rotation.to_euler(convention=convention))
    assert np.allclose(e, euler, atol=1e-3)


@pytest.mark.parametrize('rotation, improper, expected, improper_expected', [
    (
        np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1, 0, 0, 1],
        ]),
        [0, 0],
        np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.707106, 0, 0, 0.707106],
        ]),
        [0, 0]
    ),
    (
        np.array([
            [0.5, 0.5, 0.5, 0.5],
            [1, 0, 0, 1],
        ]),
        [0, 1],
        np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.707106, 0, 0, 0.707106],
        ]),
        [0, 1]
    ),
    (
        np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ]),
        [0, 0],
        np.array([
            [0.5, 0.5, 0.5, 0.5],
        ]),
        [0]
    ),
    (
        np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ]),
        [0, 1],
        np.array([
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ]),
        [0, 1]
    )
], indirect=['rotation'])
def test_unique(rotation, improper, expected, improper_expected):
    rotation.improper = improper
    u = rotation.unique
    assert np.allclose(u.data, expected, atol=1e-6)
    assert np.allclose(u.improper, improper_expected)


@pytest.mark.parametrize('rotation, improper, expected, improper_expected', [
    (
        np.array([
            [ 0.231386,  0.270835,  0.779474,  0.515294],
            [-0.515294, -0.779474,  0.270835,  0.231386],
        ]),
        [0, 1],
        np.array([
            [ 0.231386, -0.270835, -0.779474, -0.515294],
            [-0.515294,  0.779474, -0.270835, -0.231386],
        ]),
        [0, 1],
    ),
], indirect=['rotation'])
def test_inv(rotation, improper, expected, improper_expected):
    rotation.improper = improper
    r = ~rotation
    assert np.allclose(r.data, expected, atol=1e-6)
    assert np.allclose(r.improper, improper_expected)


@pytest.mark.parametrize('rotation, improper, rotation_2, improper_2, expected', [
    (
        np.array([
            [-0.192665,    -0.7385,   0.605678,   -0.22506],
            [ 0.194855, -0.0613995,   0.814759,  -0.542614],
            [-0.440859,   -0.61701,  -0.305151,   0.576042],
        ]),
        [0, 0, 0],
        np.array([
            [  0.311833,  -0.670051,  -0.635546,   -0.22332],
            [-0.0608553,  -0.380776,     -0.662,   0.642699],
        ]),
        [0, 1],
        np.array([
            [0.1001, 0],
            [0.2947, 0],
            [0.3412, 0],
        ])
    ),
    (
        np.array([
            [
                [ 0.75175,   0.250266,  -0.352737,    0.49781],
                [0.242073,  -0.698966,   0.315235,  -0.594537],
                [ 0.46822,    0.43453,  -0.653468,    0.40612],
                [0.472186,  -0.414235,  -0.552524,  -0.547875],
                [0.767081,  -0.320688,  0.0707849,   0.551122],
            ],
            [
                [-0.507603,   -0.63199,  -0.441212,   0.385045],
                [ 0.775813,   0.122649,  -0.616902, -0.0500386],
                [ 0.243256,   0.243706,   0.919676,    0.18876],
                [ 0.472742,   0.453436,   0.677063,  -0.335405],
                [0.0951788, -0.0223328,   0.924478,  -0.368487],
            ]
        ]),
        np.array([[1, 0, 0, 1, 0], [1, 1, 0, 1, 1]]),
        np.array([
            [0.733623, -0.289254,  -0.51314, -0.338846],
            [0.654535,  0.491901,  0.544886, -0.180876],
            [0.529135,  0.166796, -0.329274,  0.764051],
        ]),
        [0, 0, 1],
        np.array([
            [
               [     0,         0,    0.9360],
               [0.4195,    0.0939,         0],
               [0.4155,    0.0907,         0],
               [     0,         0,    0.0559],
               [0.4324,    0.2832,         0],
            ],
            [
               [     0,         0,    0.0655],
               [     0,         0,    0.5959],
               [0.4279,    0.7461,         0],
               [     0,         0,    0.1534],
               [     0,         0,    0.5393],
            ]
        ])
    ),
], indirect=['rotation', 'rotation_2'])
def test_dot_outer_rot(rotation, improper, rotation_2, improper_2, expected):
    rotation.improper = improper
    rotation_2.improper = improper_2
    cosines = rotation.dot_outer(rotation_2)
    assert cosines.shape == rotation.shape + rotation_2.shape
    assert np.allclose(cosines, expected, atol=1e-4)


@pytest.mark.parametrize('rotation, improper, quaternion, expected', [
    (
        np.array([
            [0.915014,  0.033423, -0.292416,  0.275909],
            [0.117797, -0.260041,  -0.54774,  0.786437],
            [0.301376,  0.818476,  0.482242, 0.0819321],
        ]),
        [0, 0, 1],
        np.array([
            [ 0.15331, -0.0110295,   -0.17113,   0.973185],
            [0.969802,   0.089686,   0.186519,   -0.12904],
        ]),
        np.array([
            [0.4585, 0.8002],
            [0.8800, 0.1127],
            [     0,      0],
        ])
    ),
], indirect=['rotation', 'quaternion'])
def test_dot_outer_quat(rotation, improper, quaternion, expected):
    rotation.improper = improper
    cosines = rotation.dot_outer(quaternion)
    assert cosines.shape == rotation.shape + quaternion.shape
    assert np.allclose(cosines, expected, atol=1e-4)