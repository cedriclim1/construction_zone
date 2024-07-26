import unittest

import numpy as np
from czone_test_fixtures import czone_TestCase
from pymatgen.core import Lattice, Structure
from test_transform import get_random_mapping

from czone.generator.generator import AmorphousGenerator, Generator, NullGenerator
from czone.transform.post import ChemicalSubstitution
from czone.transform.strain import HStrain
from czone.transform.transform import (
    Inversion,
    Reflection,
    Rotation,
    Translation,
    rot_vtv,
)
from czone.volume.algebraic import Plane, Sphere
from czone.volume.volume import Volume

seed = 72349
rng = np.random.default_rng(seed=seed)


def get_transforms():
    for basis_only in [False, True]:
        origin = rng.uniform(-10, 10, size=(1, 3))
        yield (
            Inversion(origin=origin, basis_only=basis_only),
            f"Inversion with basis_only={basis_only}",
        )

        plane = Plane(rng.normal(size=(3,)), rng.normal(size=(3,)))
        yield (
            Reflection(plane, basis_only=basis_only),
            f"Reflection with basis_only={basis_only}",
        )

        normal = rng.normal(size=(3,))
        R = rot_vtv([0, 0, 1], normal)
        yield (
            Rotation(R, origin=origin, basis_only=basis_only),
            f"Rotation with basis_only={basis_only}",
        )

        yield (
            Translation(shift=rng.uniform(-10, 10, size=(1, 3)), basis_only=basis_only),
            f"Translation with basis_only={basis_only}",
        )


def get_transformed_generators(G):
    yield G.from_generator(), " Identity"
    for t, msg in get_transforms():
        yield G.from_generator(transformation=[t]), msg


class Test_NullGenerator(czone_TestCase):
    def test_init_and_eq(self):
        A = NullGenerator()
        B = NullGenerator()
        self.assertEqual(A, B)
        self.assertReprEqual(A)

        for g, msg in get_transformed_generators(A):
            self.assertEqual(A, g, msg=f"Failed with {msg}")

    def test_supply_atoms(self):
        A = NullGenerator()
        bbox = rng.normal(size=(8, 3))
        pos, species = A.supply_atoms(bbox)
        self.assertEqual(pos.shape, (0, 3))
        self.assertEqual(species.shape, (0,))
        for g, msg in get_transformed_generators(A):
            t_pos, t_species = g.supply_atoms(bbox)
            self.assertArrayEqual(pos, t_pos, msg=f"Failed with {msg}")
            self.assertArrayEqual(species, t_species, f"Failed with {msg}")


def get_random_generator(N_species=8, with_strain=True, with_sub=True, rng=rng):
    if with_strain and rng.uniform() < 0.5:
        hstrain = HStrain(rng.uniform(size=(3,)))
    else:
        hstrain = None

    if with_sub and rng.uniform() < 0.5:
        chem_sub = ChemicalSubstitution(get_random_mapping(rng), frac=rng.uniform())
    else:
        chem_sub = None

    lattice = Lattice(5 * np.eye(3) + rng.normal(size=(3, 3)))

    N_species = rng.integers(1, N_species)
    species = rng.integers(1, 119, size=(N_species))
    pos = rng.uniform(size=(N_species, 3))

    structure = Structure(lattice, species, pos)

    origin = rng.uniform(-10, 10, size=(1, 3))

    return Generator(
        origin=origin,
        structure=structure,
        strain_field=hstrain,
        post_transform=chem_sub,
    )


class Test_Generator(czone_TestCase):
    def setUp(self):
        self.N_trials = 128

    def test_init(self):
        for _ in range(self.N_trials):
            G_0 = get_random_generator()
            for G, msg in get_transformed_generators(G_0):
                self.assertReprEqual(G, msg)

    def test_eq(self):
        for _ in range(self.N_trials):
            G = get_random_generator(with_strain=False, with_sub=False)
            H = G.from_generator()

            self.assertEqual(G, H)
            V = Volume(alg_objects=Sphere(radius=10, center=np.zeros(3)))
            bbox = V.get_bounding_box()
            gpos, gspecies = G.supply_atoms(bbox)
            hpos, hspecies = H.supply_atoms(bbox)

            self.assertArrayEqual(gpos, hpos)
            self.assertArrayEqual(gspecies, hspecies)


def get_random_amorphous_generator(rng=rng):
    origin = rng.uniform(-10, 10, size=(1, 3))
    min_dist = rng.uniform(0.5, 10)
    density = rng.uniform(0.05, 1.0)
    species = rng.integers(1, 119)

    return AmorphousGenerator(origin, min_dist, density, species)


class Test_AmorphousGenerator(czone_TestCase):
    def setUp(self):
        self.N_trials = 128

    def test_init(self):
        for _ in range(self.N_trials):
            G = get_random_amorphous_generator()
            self.assertReprEqual(G)
