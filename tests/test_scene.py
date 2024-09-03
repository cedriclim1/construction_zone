import numpy as np

from czone.molecule import Molecule
from czone.scene import PeriodicScene, Scene
from czone.volume import MultiVolume, Sphere, Volume
from czone.util.voxel import Voxel

from .czone_test_fixtures import czone_TestCase
from .test_generator import get_random_generator
from .test_volume import get_random_volume

seed = 709123
rng = np.random.default_rng(seed=seed)


def get_random_object(rng=rng, depth=0, generator_args={}):
    if depth < 2:
        vol_type = rng.choice(["Volume", "MultiVolume"])
    else:
        vol_type = "Volume"

    match vol_type:
        case "Volume":
            G = get_random_generator(rng=rng, **generator_args)
            V = get_random_volume(G, N_points=8, rng=rng)
        case "MultiVolume":
            N_vols = rng.integers(2, 8 - 3 * depth)
            Vs = [
                get_random_object(rng, depth=depth + 1, generator_args=generator_args)
                for _ in range(N_vols)
            ]
            V = MultiVolume(Vs, priority=rng.integers(-10, 10))

    return V


def get_random_domain(rng=rng):
    bases = rng.normal(size=(3, 3))
    scale = rng.uniform(0.1, 10)
    origin = rng.uniform(-100, 100, size=(3,))

    return Voxel(bases, scale, origin)


def get_random_scene(periodic=False, N_max_objects=8, rng=rng, generator_args={}):
    N_objects = rng.integers(1, N_max_objects)
    domain = get_random_domain(rng=rng)
    objects = [get_random_object(rng=rng, generator_args=generator_args) for _ in range(N_objects)]

    if periodic:
        pbc = tuple((bool(rng.choice([True, False])) for _ in range(3)))
        return PeriodicScene(domain, objects, pbc=pbc)
    else:
        return Scene(domain, objects)


class Test_Scene(czone_TestCase):
    def setUp(self):
        self.rng = rng
        self.N_trials = 32

    def test_init(self):
        for _ in range(self.N_trials):
            scene = get_random_scene()
            self.assertReprEqual(scene)

        domain = Voxel()
        ## init with one sphere
        sphere = Sphere(1, np.zeros((3, 1)))
        self.assertRaises(TypeError, lambda: Scene(domain, objects=sphere))

        ## init with two spheres
        spheres = [Volume(alg_objects=[Sphere(1, np.zeros((3, 1)))]), Sphere(5, np.zeros((3, 1)))]
        self.assertRaises(TypeError, lambda: Scene(domain, objects=spheres))

        spheres = [
            Volume(alg_objects=[Sphere(1, np.zeros((3, 1)))]),
            Volume(alg_objects=[Sphere(5, np.zeros((3, 1)))]),
        ]
        scene = Scene(domain, objects=spheres)

        # Check to see that references are carried around and not copies
        ref_ids = [id(x) for x in spheres]
        test_ids = [id(x) for x in scene.objects]
        self.assertEqual(set(ref_ids), set(test_ids))

    def test_eq(self):
        for _ in range(self.N_trials):
            ref = get_random_scene()
            test = Scene(ref.domain, objects=rng.permutation(ref.objects))
            self.assertEqual(ref, test)
            self.assertEqual(set([id(o) for o in ref.objects]), set([id(o) for o in test.objects]))

    def test_get_priorities(self):
        def get_objects(N_objects, min_priority=-10, max_priority=10):
            for _ in range(N_objects):
                sphere = Sphere(1, np.zeros((3, 1)))
                vol = Volume(
                    alg_objects=[sphere],
                    priority=int(rng.integers(min_priority, max_priority)),
                )
                yield vol

        for _ in range(self.N_trials):
            ## Get test objects and store in buckets by priority
            objs = list(get_objects(64))
            ref_dict = {}
            for o in objs:
                if o.priority in ref_dict:
                    ref_dict[o.priority].append(id(o))
                else:
                    ref_dict[o.priority] = [id(o)]

            ## Create scene and get priority array
            scene = Scene(domain=Voxel(), objects=objs)
            rel_plevels, offsets = scene._get_priorities()

            orig_priorities = [o.priority for o in objs]
            uniq_priorities = np.unique(orig_priorities)
            N_priorities = len(np.unique(orig_priorities))

            ## Check that priority array has been compressed correctly
            self.assertTrue(np.all(rel_plevels >= 0))
            self.assertTrue(len(offsets) == N_priorities + 1)

            ## Check that all objects are accounted for and counted uniquely
            for rel_p, up in enumerate(uniq_priorities):
                test_ids = [id(scene.objects[i]) for i in range(offsets[rel_p], offsets[rel_p + 1])]
                self.assertEqual(set(test_ids), set(ref_dict[up]))
                self.assertTrue(len(test_ids) == len(ref_dict[up]))

    def test_populate(self):
        def get_objects(N_objects, min_priority=-10, max_priority=10):
            for _ in range(N_objects):
                N = 32
                species = self.rng.integers(1, 119, (N, 1))
                positions = self.rng.normal(size=(N, 3))
                mol = Molecule(species, positions)
                sphere = Sphere(self.rng.uniform(2, 10), self.rng.uniform(-10, 10, (3,)))
                vol = Volume(
                    alg_objects=[sphere],
                    generator=mol,
                    priority=int(rng.integers(min_priority, max_priority)),
                )
                yield vol

        def sort_atom_arrays(atoms, species):
            order = np.argsort(atoms[:, 0])
            atoms = atoms[order]
            species = species[order]
            return atoms, species

        def get_atoms_from_scene(s):
            test_atoms, test_species = s.all_atoms, s.all_species
            return sort_atom_arrays(test_atoms, test_species)

        def get_atoms_from_objects(objs):
            ref_atoms = np.vstack([o.atoms for o in objs])
            ref_species = np.concatenate([o.species for o in objs])
            return sort_atom_arrays(ref_atoms, ref_species)

        def brute_force_collision(objs):
            atoms = []
            species = []
            for i, iobj in enumerate(objs):
                current_atoms = iobj.atoms
                current_species = iobj.species
                check = np.ones(current_atoms.shape[0], dtype=bool)
                for j, jobj in enumerate(objs):
                    if i == j:
                        continue

                    if jobj.priority <= iobj.priority:
                        check = np.logical_and(
                            check, np.logical_not(jobj.checkIfInterior(current_atoms))
                        )

                atoms.append(current_atoms[check, :])
                species.append(current_species[check])

            ref_atoms = np.vstack(atoms)
            ref_species = np.concatenate(species)
            return sort_atom_arrays(ref_atoms, ref_species)

        for _ in range(self.N_trials):
            objs = list(get_objects(32))
            scene = Scene(domain=Voxel(), objects=objs)

            ## Populate scene and sort atoms by position
            scene.populate_no_collisions()
            test_atoms, test_species = get_atoms_from_scene(scene)

            ref_atoms, ref_species = get_atoms_from_objects(objs)

            self.assertTrue(np.array_equal(test_atoms, ref_atoms))
            self.assertTrue(np.array_equal(test_species, ref_species))

            ## Check collision handling
            scene.populate()
            test_atoms, test_species = get_atoms_from_scene(scene)
            ref_atoms, ref_species = brute_force_collision(objs)
            self.assertTrue(np.array_equal(test_atoms, ref_atoms))
            self.assertTrue(np.array_equal(test_species, ref_species))

    # def test_add_object(self):
    #     ## init with two spheres
    #     spheres = [Volume(alg_objects=[Sphere(1, np.zeros((3,1)))])]
    #     scene = Scene(objects=spheres)
    #     scene.add_object(Volume(alg_objects=[Sphere(1, np.zeros((3,1)))]))


class Test_PeriodicScene(czone_TestCase):
    def setUp(self):
        self.rng = rng
        self.N_trials = 32

    def test_init(self):
        for _ in range(self.N_trials):
            scene = get_random_scene(periodic=True)
            self.assertReprEqual(scene)
