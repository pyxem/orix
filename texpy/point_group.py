import numpy as np

from texpy.vector.neo_euler import AxAngle
from texpy.quaternion.rotation import Rotation

hermann_mauguin = [
    '1', '-1',  # triclinic
    '2', 'm', '2/m',  # monoclinic
    '222', 'mm2', 'mmm',  # orthorhombic
    '4', '-4', '4/m', '422', '4mm', '-42m', '4/mmm',  # tetragonal
    '3', '-3', '32', '3m', '-3m',  # trigonal
    '6', '-6', '6/m', '622', '6mm', '-6m2', '6/mmm',  # hexagonal
    '23', 'm-3', '432', '-43m', 'm-3m',  # cubic
]
lattice = [
    'triclinic', 'triclinic',
    'monoclinic', 'monoclinic', 'monoclinic',
    'orthorhombic', 'orthorhombic', 'orthorhombic',
    'tetragonal', 'tetragonal', 'tetragonal', 'tetragonal', 'tetragonal',
    'tetragonal', 'tetragonal',
    'trigonal', 'trigonal', 'trigonal', 'trigonal', 'trigonal',
    'hexagonal', 'hexagonal', 'hexagonal', 'hexagonal', 'hexagonal',
    'hexagonal', 'hexagonal',
    'cubic', 'cubic', 'cubic', 'cubic', 'cubic'
]
proper_id = [
    1, 1,
    2, 2, 2,
    3, 3, 3,
    4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7
]
laue_class = [
    '-1', '-1',
    '2/m', '2/m', '2/m',
    'mmm', 'mmm', 'mmm',
    '4/m', '4/m', '4/m',
    '4/mmm', '4/mmm', '4/mmm', '4/mmm',
    '-3', '-3',
    '-3m', '-3m', '-3m',
    '6/m', '6/m', '6/m',
    '6/mmm', '6/mmm', '6/mmm', '6/mmm',
    'm-3', 'm-3',
    'm-3m', 'm-3m', 'm-3m',
]
inversion = [
    ((1,),), ((1,), (-1,)),  # triclinic
    ((1,),), ((-1,),), ((1,), (-1,)),  # monoclinic
    ((1, 1,),), ((-1, 1),), ((1, 1), (-1, -1)),  # orthorhombic
    ((1,),), ((-1,),), ((1,), (-1,)), ((1, 1),), ((-1, 1),), ((1, -1),), ((1, 1), (-1, -1)),  # tetragonal
    ((1,),), ((1,), (-1,)), ((1, 1),), ((-1, 1),), ((1, 1), (-1, -1)),  # trigonal
    ((1,),), ((-1,),), ((1,), (-1,)), ((1, 1),), ((-1, 1),), ((-1, -1),), ((1, 1), (-1, -1)),  # hexagonal
    ((1, 1, 1),), ((1, 1, 1), (-1, -1, -1)), ((1, 1, 1),), ((1, -1, -1),), ((1, 1, 1), (-1, -1, -1)),  # cubic
]  # TODO: this would be neater as lists

point_group_data = [
    {
        'international': i,
        'lattice': l,
        'laue_class': c,
        'inversion': np.array(j)
    } for i, l, c, j in zip(hermann_mauguin, lattice, laue_class, inversion)
]


class PointGroup:

    international = None
    lattice = None
    laue_class = None
    inversion = None

    def __init__(self, symbol=None):
        symbol = '-1' if symbol is None else symbol
        try:
            data = next(
                pg for pg in point_group_data[::-1] if symbol == pg['international'] or symbol == pg['lattice'])
        except StopIteration:
            raise ValueError("Symbol '{}' not recognised!".format(symbol))
        self.international = data['international']
        self.lattice = data['lattice']
        self.laue_class = data['laue_class']
        self.inversion = data['inversion']

    def __repr__(self):
        return 'Point Group {}'.format(self.international)

    def rotations(self, a, b, c):
        m = a - b
        axis_110 = a + b
        axis_111 = a + b + c
        axangles = {
            '2/m': [(c, 2)],
            'mmm': [(a, 2), (c, 2),],
            '4/m': [(c, 4),],
            '4/mmm': [(a, 2), (c, 4),],
            '-3': [(c, 3),],
            '-3m': [(m, 2), (c, 3),],
            '6/m': [(c, 6),],
            '6/mmm': [(a, 2), (c, 6),],
            'm-3': [(axis_111, 3), (a, 2,), (c, 2,),],
            'm-3m': [(axis_111, 3), (axis_110, 2), (c, 4),],
        }.get(self.laue_class)
        # Do rotations
        rot = []
        for ax, n in axangles:
            angles = (2 * np.pi / n) * np.arange(n)
            r = Rotation.stack(
                [Rotation.from_neo_euler(
                    AxAngle.from_axes_angles(ax, angle)) for angle in angles
                 ]).flatten()
            rot.append(r)
        if self.inversion.shape[0] == 2:
            rot.append(Rotation([[1, 0, 0, 0], [1, 0, 0, 0]]) * [1, -1])
        else:
            rot = [r * i ** np.arange(r.size) for r, i in zip(rot, self.inversion[0])]
        return rot