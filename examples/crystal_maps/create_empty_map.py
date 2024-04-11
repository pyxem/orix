"""
========================
Create empty crystal map
========================

This example shows how to create an empty crystal map of a given shape.
By empty, we mean that it is filled with identity rotations.

This crystal map can be useful for testing.
"""

from orix.crystal_map import CrystalMap

xmap = CrystalMap.empty((5, 10))

print(xmap)
print(xmap.rotations)

xmap.plot("x")
