import numpy as np
from orix.vector import Vector3d
from orix.grid.s1grid import S1Grid


class S2Grid:

    theta_grid = None  # type: S1Grid
    rho_grid = None  # type: S1Grid
    points = None  # type: Vector3d

    def __init__(self, theta_grid: S1Grid, rho_grid: S1Grid):
        self.theta_grid = theta_grid
        self.rho_grid = rho_grid
        theta = np.tile(theta_grid.points, rho_grid.points.shape)
        rho = rho_grid.points
        v = Vector3d.from_polar(theta, rho)
        self.points = v
