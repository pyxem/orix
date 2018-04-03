"""Collections of transformations representing a symmetry group.

An object's symmetry can be characterized by the transformations relating
symmetrically-equivalent views on that object. Consider the following shape.

.. image:: /_static/img/triad-object.png
   :width: 200px
   :alt: Image of an object with three-fold symmetry.
   :align: center

This obviously has three-fold symmetry. If we rotated it by
:math:`\\frac{2}{3}\\pi` or :math:`\\frac{4}{3}\\pi`, the image would be unchanged.
These angles, as well as :math:`0`, or the identity, expressed as quaternions,
form a group. Applying any operation in the group to any other results in
another member of the group.

Symmetries can consist of rotations or mirror operations, expressed as
improper rotations.

"""
import numpy as np

from texpy.quaternion.rotation import Rotation


class Symmetry(Rotation):

    @classmethod
    def generate_from(cls, *symmetries):
        generator = cls((1, 0, 0, 0))
        for symmetry in symmetries:
            generator = generator.outer(symmetry).unique()
        print(generator.improper)
        size = 1
        size_new = generator.size
        while size_new != size:
            size = size_new
            generator = generator.outer(generator).unique()
            size_new = generator.size
        return generator


C1 = Symmetry((1, 0, 0, 0))
Ci = Symmetry([(1, 0, 0, 0), (1, 0, 0, 0)]); Ci.improper = [0, 1]

C2x = Symmetry([(1, 0, 0, 0), (0, 1, 0, 0)])
C2y = Symmetry([(1, 0, 0, 0), (0, 0, 1, 0)])
C2z = Symmetry([(1, 0, 0, 0), (0, 0, 0, 1)])
C2 = Symmetry(C2z)

Csx = Symmetry([(1, 0, 0, 0), (0, 1, 0, 0)]); Csx.improper = [0, 1]
Csy = Symmetry([(1, 0, 0, 0), (0, 0, 1, 0)]); Csy.improper = [0, 1]
Csz = Symmetry([(1, 0, 0, 0), (0, 0, 0, 1)]); Csz.improper = [0, 1]
Cs = Symmetry(Csz)

C2h = Symmetry.generate_from(C2, Cs)
D2 = Symmetry.generate_from(C2z, C2x, C2y)
C2v = Symmetry.generate_from(C2z, Csx)
D2h = Symmetry.generate_from(Csz, Csx, Csy)

C4x = Symmetry([
    (1, 0, 0, 0),
    (0.5**0.5, 0.5**0.5, 0, 0),
    (0, 1, 0, 0),
    (-0.5**0.5, 0.5**0.5, 0, 0),
])
C4y = Symmetry([
    (1, 0, 0, 0),
    (0.5**0.5, 0, 0.5**0.5, 0),
    (0, 0, 1, 0),
    (-0.5**0.5, 0, 0.5**0.5, 0),
])
C4z = Symmetry([
    (1, 0, 0, 0),
    (0.5**0.5, 0, 0, 0.5**0.5),
    (0, 0, 0, 1),
    (-0.5**0.5, 0, 0, 0.5**0.5),
])
C4 = Symmetry(C4z)

S4 = Symmetry.generate_from(C2, Ci)
C4h = Symmetry.generate_from(C4, Cs)
D4 = Symmetry.generate_from(C4, C2x, C2y)
C4v = Symmetry.generate_from(C4, Csx)
D2d = Symmetry.generate_from(S4, C2x, Csy)
D4h = Symmetry.generate_from(C4h, Csx, Csy)

C3x = Symmetry([(1, 0, 0, 0), (0.5, 0.75**0.5, 0, 0), (-0.5, 0.75**0.5, 0, 0)])
C3y = Symmetry([(1, 0, 0, 0), (0.5, 0, 0.75**0.5, 0), (-0.5, 0, 0.75**0.5, 0)])
C3z = Symmetry([(1, 0, 0, 0), (0.5, 0, 0, 0.75**0.5), (-0.5, 0, 0, 0.75**0.5)])
C3 = Symmetry(C3z)

S6 = Symmetry.generate_from(C3, Ci)
D3 = Symmetry.generate_from(C3, C2x)
C3v = Symmetry.generate_from(C3, Csx)
D3d = Symmetry.generate_from(S6, Csx)

C6 = Symmetry.generate_from(C3, C2)
C3h = Symmetry.generate_from(C3, Cs)
C6h = Symmetry.generate_from(C6, Cs)
D6 = Symmetry.generate_from(C6, C2x, C2y)
C6v = Symmetry.generate_from(C6, Csx, Csy)
D3h = Symmetry.generate_from(C3h, Csx, C2y)
D6h = Symmetry.generate_from(C6h, Csx, Csy)



cubic = Rotation((0.5, 0.5, 0.5, 0.5))
T = Symmetry.generate_from(C2, cubic)
Th = Symmetry.generate_from(T, Ci)
O = Symmetry.generate_from(C4, cubic, C2x)
Td = Symmetry.generate_from(S4, cubic, Csx)
Oh = Symmetry.generate_from(O, Ci)



