"""Plotting of quaternions and related objects.

Quaternions, being four-dimensional, cannot be directly plotted. However,
various vector parametrizations of rotations can. Quaternion plots therefore
define a consistent transformation on the objects passed.

"""
import numpy as np
import matplotlib.pyplot as plt


def stereographic_transformation(x, y, z):
    assert np.allclose(x**2 + y**2 + z**2, 1)
    return x / (1 - z), y / (1 - z)


class StereographicProjection:

    def __init__(self):
        pass

    def plot(self, quaternion):
        pass

