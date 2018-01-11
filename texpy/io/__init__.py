import numpy as np


def loadang(file_string: str):
    from texpy.quaternion.rotation import Rotation
    data = np.loadtxt(file_string)
    euler = data[:, :3]
    rotation = Rotation.from_euler(euler)
    return rotation
