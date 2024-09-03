# ruff: noqa: F401

import sys
import unittest

## Import everything into the namespace so that repr's can evaluate
import numpy as np
from numpy import array
from pymatgen.core import Lattice, Structure

from czone.generator import AmorphousGenerator, Generator, NullGenerator
from czone.molecule import Molecule
from czone.scene import PeriodicScene, Scene
from czone.transform import ChemicalSubstitution, HStrain
from czone.util.voxel import Voxel
from czone.volume import Cylinder, MultiVolume, Plane, Sphere, Volume


class czone_TestCase(unittest.TestCase):
    def assertArrayEqual(self, first, second, msg=None) -> None:
        "Fail if the two arrays are unequal by via Numpy's array_equal method."
        self.assertTrue(np.array_equal(first, second), msg=msg)

    def assertReprEqual(
        self,
        obj,
        msg=None,
    ) -> None:
        "Fail if the object re-created by the __repr__ method is not equal to the original."
        with np.printoptions(threshold=sys.maxsize, floatmode="unique"):
            self.assertEqual(obj, eval(repr(obj)), msg=msg)
