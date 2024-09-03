import numpy as np

from czone.transform.post import ChemicalSubstitution
from czone.transform.strain import HStrain

from .czone_test_fixtures import czone_TestCase

seed = 871342
rng = np.random.default_rng(seed=seed)


def get_random_mapping(rng: np.random.Generator, N=8):
    Z = np.arange(1, 119, dtype=int)
    targets = rng.choice(Z, N, replace=False)
    subs = rng.choice(np.setdiff1d(Z, targets), N, replace=False)
    return {int(t): int(s) for t, s in zip(targets, subs)}


class Test_ChemicalSubstitution(czone_TestCase):
    def setUp(self):
        self.N_trials = 128

    def test_init(self):
        for _ in range(self.N_trials):
            frac = rng.uniform()
            mapping = get_random_mapping(rng)
            chem_sub = ChemicalSubstitution(mapping, frac)
            self.assertReprEqual(chem_sub)


class Test_HStrain(czone_TestCase):
    def setUp(self):
        self.N_trials = 128

    def test_init(self):
        def yield_strain_matrices():
            for shape in ((3,), (3, 3), (9,), (6,)):
                yield rng.uniform(size=shape)

        for _ in range(self.N_trials):
            for m in yield_strain_matrices():
                hstrain = HStrain(m, origin="generator", mode="crystal")
                self.assertReprEqual(hstrain)
