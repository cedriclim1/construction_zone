from pathlib import Path

import numpy as np

from czone.blueprint import Blueprint, Serializer
from czone.generator import NullGenerator
from czone.util.voxel import Voxel

from .czone_test_fixtures import czone_TestCase
from .test_generator import get_random_amorphous_generator, get_random_generator
from .test_scene import get_random_object, get_random_scene

seed = 9815108923
rng = np.random.default_rng(seed=seed)


class Test_Blueprint(czone_TestCase):
    """object -> blueprint -> object"""

    def setUp(self):
        self.N_trials = 32

    def test_generator(self):
        n_G = NullGenerator()
        n_blueprint = Blueprint(n_G)
        self.assertEqual(n_G, n_blueprint.to_object())
        self.assertReprEqual(n_G)

        for _ in range(self.N_trials):
            a_G = get_random_amorphous_generator(rng=rng)
            a_blueprint = Blueprint(a_G)
            self.assertReprEqual(a_G)
            self.assertEqual(a_G, a_blueprint.to_object())

            G = get_random_generator(rng=rng)
            blueprint = Blueprint(G)
            self.assertReprEqual(G)
            self.assertEqual(G, blueprint.to_object())

    def test_volume(self):
        for _ in range(self.N_trials):
            V = get_random_object()
            blueprint = Blueprint(V)
            self.assertEqual(V, blueprint.to_object())

    def test_voxel(self):
        for _ in range(self.N_trials):
            bases = rng.normal(size=(3, 3))
            scale = rng.uniform(0.1, 10)
            origin = rng.uniform(-100, 100, size=(3,))

            V = Voxel(bases, scale, origin)
            blueprint = Blueprint(V)
            self.assertEqual(V, blueprint.to_object())

    def test_scene(self):
        for _ in range(self.N_trials):
            for periodic in [False, True]:
                S = get_random_scene(periodic=periodic)
                blueprint = Blueprint(S)
                self.assertEqual(S, blueprint.to_object())


class Test_Serializer(czone_TestCase):
    """blueprint -> serialized form -> blueprint"""

    def setUp(self):
        self.N_trials = 16
        self.formats = ["json", "yaml", "toml", "h5"]
        self.generator_args = {"with_strain": False, "with_sub": False}

    def test_generator(self):
        for _ in range(self.N_trials):
            G = get_random_generator(**self.generator_args)
            blueprint = Blueprint(G)
            for f in self.formats:
                test_path = "generator_test_file" + "." + f
                Serializer.write(Path(test_path), blueprint)

                test_bp = Serializer.read(Path(test_path))
                test_G = test_bp.to_object()
                self.assertEqual(G, test_G)

    def test_volume(self):
        for _ in range(self.N_trials):
            V = get_random_object(generator_args=self.generator_args)
            blueprint = Blueprint(V)
            for f in self.formats:
                test_path = "volume_test_file" + "." + f
                Serializer.write(Path(test_path), blueprint)

                test_bp = Serializer.read(Path(test_path))
                test_V = test_bp.to_object()
                self.assertEqual(V, test_V)

    def test_scene(self):
        for _ in range(self.N_trials):
            for periodic in [False, True]:
                S = get_random_scene(periodic=periodic, generator_args=self.generator_args)
                blueprint = Blueprint(S)
                for f in self.formats:
                    test_path = "scene_test_file" + "." + f
                    Serializer.write(Path(test_path), blueprint)

                    test_S = Serializer.read(Path(test_path)).to_object()
                    self.assertEqual(S, test_S)
