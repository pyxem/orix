import pytest
import numpy as np
import itertools

from texpy.vector.axangle import AxAngle
from texpy.vector.vector3d import Vector3d
from texpy.quaternion.rotation import Rotation


axes = [
    (1, 0, 0),
    (1, 1, 0),
    (2, -1, 0),
    (0, 2, 1),
    (-1, -1, -1),
]

angles = [
    - 2 * np.pi,
    - 5 * np.pi / 6,
    - np.pi / 3,
    0,
    np.pi / 12,
    np.pi / 3,
    3 * np.pi / 4,
    2 * np.pi,
    np.pi/7,
]

axangles = [np.array(angle) * Vector3d(axis).unit for axis in axes for angle in
            angles]
axangles += [np.array(angle) * Vector3d(axis).unit for axis in
             itertools.combinations_with_replacement(axes, 2) for angle in
             itertools.combinations_with_replacement(angles, 2)]


@pytest.fixture(params=axangles[:100])
def axangle(request):
    return AxAngle(request.param.data)


def test_angle(axangle):
    assert np.allclose(axangle.angle, axangle.norm)


def test_axis(axangle):
    assert axangle.axis.shape == axangle.shape
    assert np.allclose(axangle.axis.norm, 1)


def test_to_rotation(axangle):
    r = axangle.to_rotation()
    assert isinstance(r, Rotation)
    assert np.allclose(r.norm, 1)
    assert np.allclose(r.angle, axangle.angle % (2*np.pi))


@pytest.mark.parametrize('axis, angle, expected_axis', [
    ((2, 1, 1), np.pi/4, (0.816496, 0.408248, 0.408248)),
    (Vector3d((2, 0, 0)), -2 * np.pi, (-1, 0, 0))
])
def test_from_axes_angles(axis, angle, expected_axis):
    ax = AxAngle.from_axes_angles(axis, angle)
    assert np.allclose(ax.axis.data, expected_axis)
    assert np.allclose(ax.angle, abs(angle))
