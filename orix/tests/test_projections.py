from orix.projections.stereographic_projection import StereographicProjection
import pytest
from orix.vector.vector3d import Vector3d
import numpy as np


@pytest.fixture()
def vector3d():
    return Vector3d(np.array(
            [
             [0, 0, 1],
             [1, 0, 0],
             [0, 1, 0],
             [0, 0, -1],
             [-1, 0, 0],
             [0, -1, 0],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 0],
             [0, -1, -1],
             [-1, 0, -1],
             [-1, -1, 0],
             [0, -1, 1],
             [-1, 0, 1],
             [-1, 1, 0],
             [0, 1, -1],
             [1, 0, -1],
             [1, -1, 0],
             [1, 1, 1],
             [-1, 1, 1],
             [1, -1, 1],
             [1, 1, -1],
             [-1, -1, 1],
             [1, -1, -1],
             [-1, 1, -1],
             [-1, -1, -1],
             ]
        ))


@pytest.fixture()
def xy():
    return np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [0, -1],
                [-1, 0],
                [0.5, 0.5],
                [-0.5, 0.5],
                [0.5, -0.5],
                [-0.5, -0.5],
            ]
            )


@pytest.fixture()
def spherical_coordinates(vector3d):
    theta, phi, _ = vector3d.to_polar()
    return theta.data, phi.data


@pytest.mark.parametrize("pole", [-1, 1])
def test_project(pole, vector3d):
    StereographicProjection.project(vector3d, pole)


@pytest.mark.xfail(raises=ValueError)
def test_project_fail(vector3d):
    StereographicProjection.project(vector3d, 2)


@pytest.mark.parametrize("pole", [-1, 1])
def test_project_spherical(pole, spherical_coordinates):
    theta, phi = spherical_coordinates
    StereographicProjection.project_spherical(theta, phi, pole=pole)


def test_project_split(vector3d):
    StereographicProjection.project_split(vector3d)


def test_project_split_spherical(spherical_coordinates):
    theta, phi = spherical_coordinates
    StereographicProjection.project_split_spherical(theta, phi)


@pytest.mark.parametrize("pole", [-1, 1])
def test_iproject(pole, xy):
    StereographicProjection.iproject(xy, pole)


@pytest.mark.parametrize("pole", [-1, 1])
def test_iproject_onearray(pole):
    data = np.array([0.5, -0.5])
    StereographicProjection.iproject(data, pole)


@pytest.mark.parametrize("pole", [-1, 1])
def test_iproject_spherical(pole, xy):
    StereographicProjection.iproject_spherical(xy, pole=pole)


def test_verify(vector3d):
    vector3d = vector3d.unit
    vector3d_up = vector3d[vector3d.z >= 0]
    vector3d_down = vector3d[vector3d.z < 0]
    stereocoords_u = StereographicProjection.project(vector3d_up, -1)
    stereocoords_d = StereographicProjection.project(vector3d_down, 1)
    inv_u = StereographicProjection.iproject(stereocoords_u, -1)
    inv_d = StereographicProjection.iproject(stereocoords_d, 1)
    np.testing.assert_allclose(inv_u.data, vector3d_up.data, rtol=1e-5, atol=1e-5)
    print(vector3d_down)
    print(inv_d)
    np.testing.assert_allclose(inv_d.data, vector3d_down.data, rtol=1e-5, atol=1e-5)


def test_verify_spherical(spherical_coordinates):
    theta, phi = spherical_coordinates
    tu = theta[theta<=np.pi/2]
    td = theta[theta>np.pi/2]
    pu = phi[theta<=np.pi/2]
    pd = phi[theta>np.pi/2]
    xyu = StereographicProjection.project_spherical(tu, pu, pole=-1)
    xyd = StereographicProjection.project_spherical(td, pd, pole=1)
    tut, put = StereographicProjection.iproject_spherical(xyu, pole=-1)
    tdt, pdt = StereographicProjection.iproject_spherical(xyd, pole=1)
    np.testing.assert_allclose(tu.data, tut.data)
    np.testing.assert_allclose(td.data, tdt.data)
    np.testing.assert_allclose(pu.data, put.data)
    np.testing.assert_allclose(pd.data, pdt.data)
