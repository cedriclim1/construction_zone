import numpy as np

from czone.util.voxel import Voxel
from czone.util.eset import EqualSet, array_set_equal

from .czone_test_fixtures import czone_TestCase

seed = 9817924
rng = np.random.default_rng(seed=seed)


class Test_EqualSet(czone_TestCase):
    def setUp(self):
        self.N_trials = 64
        self.N_points = 256

    def assertSetEquivalent(self, first, second, msg=None) -> None:
        "Fail if two sets are not equal, checking"
        self.assertEqual(len(first), len(second), msg)
        self.assertEqual(first.union(second), first, msg)
        self.assertEqual(first.symmetric_difference(second), set([]), msg)

    def test_init(self):
        for _ in range(self.N_trials):
            x = rng.integers(0, 100, size=(self.N_points))

            sx_ref = set([i for i in x])

            sx_test = EqualSet(x)
            self.assertSetEquivalent(sx_ref, sx_test)

    def test_union(self):
        for _ in range(self.N_trials):
            x = rng.integers(0, 100, size=(self.N_points))
            y = rng.integers(0, 100, size=(self.N_points))

            sx_ref = set([i for i in x]).union([j for j in y])
            sx_test = EqualSet(x).union([j for j in y])
            sx_itest = EqualSet(x)
            sx_itest |= [j for j in y]

            self.assertSetEquivalent(sx_ref, sx_test)
            self.assertSetEquivalent(sx_ref, sx_itest)
            self.assertSetEquivalent(sx_ref, EqualSet(x) | [j for j in y])

    def test_difference(self):
        for _ in range(self.N_trials):
            x = rng.integers(0, 100, size=(self.N_points))
            y = rng.integers(0, 100, size=(self.N_points))

            sx_ref = set([i for i in x]).difference([j for j in y])
            sx_test = EqualSet(x).difference([j for j in y])
            sx_itest = EqualSet(x)
            sx_itest -= [j for j in y]

            self.assertSetEquivalent(sx_ref, sx_test)
            self.assertSetEquivalent(sx_ref, sx_itest)
            self.assertSetEquivalent(sx_ref, EqualSet(x) - [j for j in y])

    def test_symmetric_difference(self):
        for _ in range(self.N_trials):
            x = rng.integers(0, 100, size=(self.N_points))
            y = rng.integers(0, 100, size=(self.N_points))

            sx_ref = set([i for i in x]).symmetric_difference([j for j in y])
            sx_test = EqualSet(x).symmetric_difference([j for j in y])

            self.assertSetEquivalent(sx_ref, sx_test)
            self.assertSetEquivalent(sx_ref, EqualSet(x) ^ [j for j in y])

    def test_intersection(self):
        for _ in range(self.N_trials):
            x = rng.integers(0, 100, size=(self.N_points))
            y = rng.integers(0, 100, size=(self.N_points))

            sx_ref = set([i for i in x]).intersection([j for j in y])
            sx_test = EqualSet(x).intersection([j for j in y])

            self.assertSetEquivalent(sx_ref, sx_test)
            self.assertSetEquivalent(sx_ref, EqualSet(x) & [j for j in y])

    def test_isdisjoint(self):
        for _ in range(self.N_trials):
            x = rng.integers(0, 100, size=(self.N_points))
            y = rng.integers(0, 100, size=(self.N_points))

            ref_A = set([i for i in x])
            ref_B = set([j for j in y])
            ref_A.difference_update(ref_B)

            true_check = EqualSet(list(ref_A)).isdisjoint(list(ref_B))
            self.assertTrue(ref_A.isdisjoint(ref_B))
            self.assertTrue(true_check)

            ref_A.update(ref_B)
            false_check = EqualSet(list(ref_A)).isdisjoint(list(ref_B))
            self.assertFalse(ref_A.isdisjoint(ref_B))
            self.assertFalse(false_check)

    def test_equalities(self):
        ## Make sure empty sets are handled correctly
        self.assertTrue(EqualSet() == EqualSet())
        self.assertTrue(EqualSet() < EqualSet([1]))
        self.assertTrue(EqualSet([1]) > EqualSet())

        for _ in range(self.N_trials):
            x = rng.integers(0, 100, size=(self.N_points))
            y = rng.integers(0, 100, size=(self.N_points))

            sx_ref = set([i for i in x])
            sy_ref = set([j for j in y])
            sx_test = EqualSet(x)
            sy_test = EqualSet(y)

            ## subset, superset
            self.assertEqual(sx_ref.issubset(sy_ref), sx_test.issubset(sy_test))
            self.assertEqual(sx_ref.issuperset(sy_ref), sx_test.issuperset(sy_test))
            self.assertEqual(sx_ref <= sy_ref, sx_test <= sy_test)
            self.assertEqual(sx_ref >= sy_ref, sx_test >= sy_test)

            ## subset AND superset
            self.assertEqual(sx_ref == sy_ref, sx_test == sy_test)
            self.assertTrue(sx_test == EqualSet(x))

            ## proper subset/supersets
            self.assertTrue((sx_test | sy_test) > sx_test)
            self.assertTrue((sx_test | sy_test) > sy_test)
            self.assertTrue(sx_test < (sx_test | sy_test))
            self.assertTrue(sy_test < (sx_test | sy_test))


class Test_ArraySet(czone_TestCase):
    def setUp(self):
        self.N_trials = 1024
        self.N_points = 1024

    def test_array_set_equal(self):
        for _ in range(self.N_trials):
            X = rng.uniform(size=(self.N_points, 3))
            Xp = rng.permutation(X, axis=0)
            self.assertTrue(array_set_equal(X, Xp))

            Xp[23, :] = rng.uniform(size=(1, 3)) + 1.0
            self.assertFalse(array_set_equal(X, Xp))


#### Tests for Voxel class
class Test_Voxel(czone_TestCase):
    def setUp(self):
        self.N_trials = 32

    def test_init(self):
        for _ in range(self.N_trials):
            bases = rng.normal(size=(3, 3))
            scale = rng.uniform(0.1, 10)
            origin = rng.uniform(-100, 100, size=(3,))

            voxel = Voxel(bases, scale, origin)
            self.assertReprEqual(voxel)
