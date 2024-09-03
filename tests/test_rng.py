import numpy as np
import pytest
from .czone_test_fixtures import czone_TestCase

from czone.generator.amorphous_algorithms import gen_p_substrate, gen_p_substrate_batched
from czone.generator.generator import AmorphousGenerator, Generator
from czone.molecule.molecule import Molecule
from czone.prefab.prefab import fccMixedTwinSF, wurtziteStackingFault
from czone.surface.adsorbate import add_adsorbate, find_approximate_normal
from czone.surface.alpha_shape import alpha_shape_alg_3D_with_sampling
from czone.transform.post import ChemicalSubstitution
from czone.util.misc import get_N_splits
from czone.volume.algebraic import Sphere
from czone.volume.volume import Volume

"""
These unit tests are not meant to measure code functionality/correctness.
Instead, these are meant to check that any method using RNGs can be
completely reproduced by passing the RNG in as a property/argument.
"""

seed = 9871492
base_rng = np.random.default_rng(seed=seed)


class Test_Functions(czone_TestCase):
    def setUp(self):
        self.N_trials = 32

    def assertConsistent(self, F, args, seed):
        # seed rng and get reference result
        rng = np.random.default_rng(seed)
        ref_state = rng.bit_generator.state  # cache state to reseed rng
        ref_res = F(*args, rng=rng)

        # reset RNG state and call function again
        rng.bit_generator.state = ref_state
        test_res = F(*args, rng=rng)

        match test_res:
            case np.ndarray():
                self.assertArrayEqual(ref_res, test_res)
            case (Molecule(), bool()):
                self.assertArrayEqual(ref_res[0].atoms, test_res[0].atoms)
                self.assertArrayEqual(ref_res[0].species, test_res[0].species)
                self.assertEqual(ref_res[1], test_res[1])
            case _:
                self.assertEqual(ref_res, test_res)

    def test_get_N_splits(self):
        L = 32
        N = 4
        M = 2
        for _ in range(self.N_trials):
            seed = base_rng.integers(0, int(1e6))
            self.assertConsistent(get_N_splits, (N, M, L), seed)

    def test_find_approximate_normal(self):
        points = base_rng.normal(size=(100, 3))
        z_filter = points[:, 2] <= 0
        test_points = points[z_filter, :]
        for _ in range(self.N_trials):
            seed = base_rng.integers(0, int(1e6))
            self.assertConsistent(find_approximate_normal, (test_points,), seed)

    @pytest.mark.filterwarnings(
        "ignore:Requested to transform molecule", "ignore:Input Volume has not pop"
    )
    def test_add_adsorbate(self):
        ## Initialize test volume
        surface_atoms = 10 * base_rng.normal(size=(256, 3))
        surface_species = np.ones(surface_atoms.shape[0])

        test_volume = Volume(
            alg_objects=Sphere(10, np.zeros(3)), generator=Molecule(surface_species, surface_atoms)
        )

        test_adsorbate = Molecule([1], np.zeros((1, 3)))

        test_args = (test_adsorbate, 0, 1.0, test_volume)

        for _ in range(self.N_trials // 2):
            seed = base_rng.integers(0, int(1e6))
            self.assertConsistent(add_adsorbate, test_args, seed)

    def test_alpha_shape_alg_3D_with_sampling(self):
        test_points = 10 * base_rng.normal(size=(256, 3))
        probe_radius = 1.0
        N_samples = 4

        for _ in range(self.N_trials):
            seed = base_rng.integers(0, int(1e6))
            self.assertConsistent(
                alpha_shape_alg_3D_with_sampling, (test_points, probe_radius, N_samples), seed
            )

    def test_gen_p_substrate(self):
        dims = (15, 15, 15)

        for _ in range(self.N_trials):
            seed = base_rng.integers(0, int(1e6))
            self.assertConsistent(gen_p_substrate, (dims,), seed)

    def test_gen_p_substrate_batched(self):
        dims = (15, 15, 15)

        for _ in range(self.N_trials):
            seed = base_rng.integers(0, int(1e6))
            self.assertConsistent(gen_p_substrate_batched, (dims,), seed)


class Test_Classes(czone_TestCase):
    def test_AmorphousGenerator(self):
        N_trials = 10
        for _ in range(N_trials):
            seed = base_rng.integers(0, int(1e6))
            rng = np.random.default_rng(seed=seed)
            ref_volume = Volume(
                alg_objects=Sphere(10, np.zeros(3)), generator=AmorphousGenerator(rng=rng)
            )

            test_volume = ref_volume.from_volume()

            self.assertNotEqual(id(ref_volume.generator.rng), id(test_volume.generator.rng))
            self.assertEqual(
                ref_volume.generator.rng.bit_generator.state,
                test_volume.generator.rng.bit_generator.state,
            )

            ref_volume.populate_atoms()
            test_volume.populate_atoms()

            self.assertArrayEqual(ref_volume.atoms, test_volume.atoms)
            self.assertArrayEqual(ref_volume.species, test_volume.species)

    def test_ChemicalSubstitution(self):
        test_generator = Generator.from_spacegroup(
            [6], np.zeros((1, 3)), [1, 1, 1], [90, 90, 90], sgn=225
        )
        ref_volume = Volume(alg_objects=Sphere(10, np.zeros(3)), generator=test_generator)
        test_volume = ref_volume.from_volume()

        N_trials = 32
        for _ in range(N_trials):
            seed = base_rng.integers(0, int(1e6))
            ref_rng = np.random.default_rng(seed=seed)
            test_rng = np.random.default_rng(seed=seed)
            ref_volume.generator.post_transform = ChemicalSubstitution({6: 8}, 0.1, rng=ref_rng)
            test_volume.generator.post_transform = ChemicalSubstitution({6: 8}, 0.1, rng=test_rng)

            ref_volume.populate_atoms()
            test_volume.populate_atoms()
            self.assertArrayEqual(ref_volume.species, test_volume.species)

    def test_fccMixedTwinSF(self):
        generator = Generator.from_spacegroup(
            [6], np.zeros((1, 3)), [2, 2, 2], [90, 90, 90], sgn=225
        )
        volume = Volume(alg_objects=Sphere(10, np.zeros(3)))

        N_trials = 32
        for _ in range(N_trials):
            seed = base_rng.integers(0, int(1e6))
            ref_rng = np.random.default_rng(seed=seed)
            test_rng = np.random.default_rng(seed=seed)

            ref_fab = fccMixedTwinSF(generator, volume, rng=ref_rng)
            test_fab = fccMixedTwinSF(generator, volume, rng=test_rng)

            ref_obj = ref_fab.build_object()
            ref_obj.populate_atoms()

            test_obj = test_fab.build_object()
            test_obj.populate_atoms()

            self.assertArrayEqual(ref_obj.atoms, test_obj.atoms)
            self.assertArrayEqual(ref_obj.species, test_obj.species)

    def test_wurtziteStackingFault(self):
        generator = Generator.from_spacegroup(
            [6], np.zeros((1, 3)), [2, 2, 3], [90, 90, 120], sgn=186
        )
        volume = Volume(alg_objects=Sphere(10, np.zeros(3)))

        N_trials = 32
        for _ in range(N_trials):
            seed = base_rng.integers(0, int(1e6))
            ref_rng = np.random.default_rng(seed=seed)
            test_rng = np.random.default_rng(seed=seed)

            ref_fab = wurtziteStackingFault(generator, volume, rng=ref_rng)
            test_fab = wurtziteStackingFault(generator, volume, rng=test_rng)

            ref_obj = ref_fab.build_object()
            ref_obj.populate_atoms()

            test_obj = test_fab.build_object()
            test_obj.populate_atoms()

            self.assertArrayEqual(ref_obj.atoms, test_obj.atoms)
            self.assertArrayEqual(ref_obj.species, test_obj.species)
