from functools import reduce

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from czone.generator import Generator, NullGenerator
from czone.transform import Rotation
from czone.util.eset import EqualSet, array_set_equal
from czone.volume import (
    Cylinder,
    MultiVolume,
    Plane,
    Sphere,
    Volume,
    convex_hull_to_planes,
    get_bounding_box,
)

from .czone_test_fixtures import czone_TestCase

seed = 32135213035
rng = np.random.default_rng(seed=seed)

#### Tests for Algebraic Objects


class Test_Sphere(czone_TestCase):
    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        for _ in range(self.N_trials):
            radius = rng.uniform(size=(1,))[0]
            center = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            sphere = Sphere(radius, center, tol=tol)
            self.assertReprEqual(sphere)

            self.assertEqual(radius, sphere.radius)
            self.assertTrue(np.array_equal(center, sphere.center))
            self.assertEqual(tol, sphere.tol)

            rt, ct = sphere.params
            self.assertEqual(radius, rt)
            self.assertTrue(np.array_equal(center, ct))

        self.assertRaises(ValueError, lambda x: Sphere(x, [0, 0, 0]), 0.0)
        self.assertRaises(ValueError, lambda x: Sphere(x, [0, 0, 0]), -1.0)
        self.assertRaises(ValueError, lambda x, y: Sphere(x, y), 1.0, [2, 0])
        self.assertRaises(
            ValueError,
            lambda x, y: Sphere(x, y),
            1.0,
            [
                [2, 0, 0],
                [
                    1,
                    0,
                    0,
                ],
            ],
        )

    def test_check_interior(self):
        for _ in range(self.N_trials):
            radius = rng.uniform(1, 1000, size=(1,))[0]
            center = rng.normal(0, 10, size=(3,))
            tol = rng.uniform(size=(1,))[0]

            sphere = Sphere(radius, center, tol=tol)

            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))
            ref_check = np.linalg.norm(test_points - center, axis=1) <= radius + tol

            self.assertTrue(np.array_equal(ref_check, sphere.checkIfInterior(test_points)))


class Test_Plane(czone_TestCase):
    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        def is_collinear(a, b):
            cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return np.isclose(cos_theta, 1.0)

        for _ in range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)
            self.assertReprEqual(plane)

            self.assertTrue(np.isclose(1.0, np.linalg.norm(plane.normal)))
            self.assertTrue(is_collinear(normal, plane.normal))
            self.assertTrue(np.array_equal(point, plane.point))
            self.assertEqual(tol, plane.tol)

            nt, pt = plane.params
            self.assertTrue(np.array_equal(point, pt))
            self.assertTrue(np.isclose(1.0, np.linalg.norm(nt)))
            self.assertTrue(is_collinear(normal, nt))

            plane.flip_orientation()
            self.assertTrue(np.isclose(1.0, np.linalg.norm(plane.normal)))
            self.assertTrue(is_collinear(-normal, plane.normal))

        self.assertRaises(ValueError, lambda x: Plane(x, (0, 0, 0)), (0, 0, 0))

    def test_eq(self):
        for _ in range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))

            plane = Plane(normal, point, tol=0.0)

            new_point = plane.project_point(rng.normal(size=(3,)))
            self.assertEqual(plane, test_plane := Plane(normal, new_point, tol=0.0))

            Rs = rng.normal(size=(3, 3))
            R = Rotation(np.linalg.qr(Rs)[0], origin=point)

            R.applyTransformation_alg(plane)
            R.applyTransformation_alg(test_plane)
            self.assertEqual(plane, test_plane)

    def test_check_interior(self):
        for _ in range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)

            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))
            ref_check = np.dot(test_points - point, normal / np.linalg.norm(normal)) < (tol)

            self.assertTrue(np.array_equal(ref_check, plane.checkIfInterior(test_points)))

            # This would fail for any points on the plane or within tolerance of the interiority check
            tol_filter = plane.dist_from_plane(test_points) > plane.tol
            test_points = test_points[tol_filter, :]

            ref_check = np.dot(test_points - point, normal / np.linalg.norm(normal)) < (tol)
            ref_arr = np.logical_not(ref_check)
            self.assertTrue(
                np.array_equal(ref_arr, plane.flip_orientation().checkIfInterior(test_points))
            )

    def test_dist_from_plane(self):
        for _ in range(self.N_trials):
            point = rng.normal(size=(3,))
            plane = Plane([0, 0, 1], point)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))

            ref_arr = np.abs(test_points[:, 2] - point[2])
            self.assertTrue(np.allclose(plane.dist_from_plane(test_points), ref_arr))

            Rs = rng.normal(size=(3, 3))
            R = Rotation(np.linalg.qr(Rs)[0], origin=point)

            r_plane = Plane.from_alg_object(plane, transformation=[R])
            r_points = R.applyTransformation(test_points)
            self.assertTrue(np.allclose(r_plane.dist_from_plane(r_points), ref_arr))

            r_plane.flip_orientation()
            self.assertTrue(np.allclose(r_plane.dist_from_plane(r_points), ref_arr))

    def test_project_point(self):
        for _ in range(self.N_trials):
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            tol = rng.uniform(size=(1,))[0]

            plane = Plane(normal, point, tol=tol)
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))

            proj_points = plane.project_point(test_points)
            proj_dist = plane.dist_from_plane(proj_points)
            self.assertTrue(np.allclose(proj_dist, np.zeros_like(proj_dist)))

    def test_convex_hull_to_planes(self):
        for _ in range(self.N_trials):
            hull_points = rng.uniform(-5, 5, size=(32, 3))
            test_points = rng.uniform(-10, 10, size=(self.N_test_points, 3))

            tri = Delaunay(hull_points)

            planes = convex_hull_to_planes(hull_points)

            ref_check = tri.find_simplex(test_points) > -1
            test_check = reduce(
                lambda x, y: np.logical_and(x, y), [p.checkIfInterior(test_points) for p in planes]
            )

            self.assertTrue(np.array_equal(test_check, ref_check))

    def test_bounding_box(self):
        for _ in range(self.N_trials):
            hull_points = rng.uniform(-5, 5, size=(32, 3))
            test_points = rng.uniform(-10, 10, size=(self.N_test_points, 3))

            tri = Delaunay(hull_points)
            planes = convex_hull_to_planes(hull_points)

            test_points, status = get_bounding_box(planes)
            ref_points = tri.points[np.unique(tri.convex_hull), :]
            self.assertTrue(array_set_equal(test_points, ref_points))

            # If < 4 planes, should return unbounded or null
            _, status = get_bounding_box(planes[:3])
            self.assertTrue(status in (2, 3))

        # If two planes mutually exclude eachother, feasible region should be null
        new_plane = Plane.from_alg_object(planes[0])
        new_plane.flip_orientation()
        new_plane.point = new_plane.point - new_plane.normal
        hs, status = get_bounding_box(planes + [new_plane])
        self.assertEqual(status, 2)

        # This raises a QHullerror, since the planes are on top of each other.
        new_plane = Plane.from_alg_object(planes[0])
        new_plane.flip_orientation()
        _, status = get_bounding_box(planes + [new_plane])
        self.assertEqual(status, 2)


class Test_Cylinder(czone_TestCase):
    def setUp(self):
        self.N_trials = 100
        self.N_test_points = 1024

    def test_init(self):
        def is_collinear(a, b):
            cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return np.isclose(cos_theta, 1.0)

        for _ in range(self.N_trials):
            axis = rng.normal(size=(3, 1))

            point = rng.normal(size=(3, 1))

            radius = rng.uniform(1e-1, 1e2)
            length = rng.uniform(1e-1, 1e2)

            cyl = Cylinder(axis, point, radius, length)
            self.assertReprEqual(cyl)

            ta, tp, tr, tl = cyl.params()

            self.assertEqual(radius, tr)
            self.assertEqual(length, tl)
            self.assertTrue(np.array_equal(point.reshape((3,)), tp))
            self.assertTrue(is_collinear(axis[:, 0], ta))
            self.assertTrue(np.isclose(np.linalg.norm(ta), 1.0))

        # good values
        ga = np.array([0, 0, 1])
        gp = np.array([0, 0, 0])
        gr = 1.0
        gl = 1.0

        ## errors on radius
        # type
        self.assertRaises(ValueError, lambda: Cylinder(ga, gp, "a", gl))

        # value
        self.assertRaises(ValueError, lambda: Cylinder(ga, gp, -1.0, gl))
        self.assertRaises(ValueError, lambda: Cylinder(ga, gp, 0.5 * np.finfo(float).eps, gl))

        ## errors on length
        # type
        self.assertRaises(ValueError, lambda: Cylinder(ga, gp, gr, "a"))

        # value
        self.assertRaises(ValueError, lambda: Cylinder(ga, gp, gr, -1.0))
        self.assertRaises(ValueError, lambda: Cylinder(ga, gp, gr, 0.5 * np.finfo(float).eps))

        ## errors on axis
        # shape
        self.assertRaises(ValueError, lambda: Cylinder([0, 0, 0, 1], gp, gr, gl))

        # value
        self.assertRaises(ValueError, lambda: Cylinder([0, 0, 0], gp, gr, gl))

        ## errors on point
        # shape
        self.assertRaises(ValueError, lambda: Cylinder(ga, [0, 0, 0, 1], gr, gl))

    def test_bounding_box(self):
        def is_collinear(a, b):
            cos_theta = np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))
            return np.all(np.logical_or(np.isclose(cos_theta, 1.0), np.isclose(cos_theta, -1.0)))

        for _ in range(self.N_trials):
            axis = rng.normal(size=(3, 1))
            point = rng.normal(size=(3, 1))
            radius = rng.uniform(1e-1, 1e2)
            length = rng.uniform(1e-1, 1e2)
            cyl = Cylinder(axis, point, radius, length)

            bbox = cyl.get_bounding_box()

            ## Assert shape
            self.assertEqual(bbox.shape, (8, 3))

            ## Box is centered correctly
            self.assertTrue(np.allclose(np.mean(bbox, axis=0), point.reshape((3,))))

            ## Box has the right length and is oriented correctly
            dvecs = bbox[:4, :] - bbox[4:, :]
            d = np.linalg.norm(dvecs, axis=1)
            self.assertTrue(np.allclose(d, length))
            self.assertTrue(is_collinear(dvecs, axis))

            ## Box has the right volume
            vol = ConvexHull(bbox).volume
            self.assertTrue(np.isclose(vol / (length * (2 * radius) ** 2.0), 1.0))

    def test_check_interior(self):
        for _ in range(self.N_trials):
            test_points = rng.uniform(-1000, 1000, size=(self.N_test_points, 3))

            # Create cylinder oriented in +-Z on XY
            axis = np.array([0, 0, 1])
            point = rng.normal(size=(1, 3))
            radius = rng.uniform(1e-1, 1e2)
            length = rng.uniform(1e-1, 1e2)
            cyl = Cylinder(axis, point, radius, length)

            ref_check_length = np.abs(test_points[:, 2] - point[:, 2]) < length / 2.0 + cyl.tol
            ref_check_radius = (
                np.linalg.norm(point[:, :2] - test_points[:, :2], axis=1) < radius + cyl.tol
            )
            ref_check = np.logical_and(ref_check_length, ref_check_radius)

            self.assertTrue(np.array_equal(ref_check, cyl.checkIfInterior(test_points)))

            # Rotate cylinder and points to random axis and assert check is equivalent
            Rs = rng.normal(size=(3, 3))
            R = Rotation(np.linalg.qr(Rs)[0], origin=point)

            r_cyl = Cylinder.from_alg_object(cyl, transformation=[R])
            r_points = R.applyTransformation(test_points)

            self.assertTrue(np.array_equal(ref_check, r_cyl.checkIfInterior(r_points)))


#### Tests for Volume and MultiVolue


def get_random_alg_object():
    choice = rng.choice(["p", "s", "c"])
    tol = rng.uniform(size=(1,))[0]
    match choice:
        case "p":
            normal = rng.normal(size=(3,))
            point = rng.normal(size=(3,))
            return Plane(normal, point, tol=tol)
        case "s":
            radius = rng.uniform(size=(1,))[0]
            center = rng.normal(size=(3,))
            return Sphere(radius, center, tol=tol)
        case "c":
            axis = rng.normal(size=(3, 1))
            point = rng.normal(size=(3, 1))
            radius = rng.uniform(1e-1, 1e2)
            length = rng.uniform(1e-1, 1e2)
            return Cylinder(axis, point, radius, length)


def get_random_volume(generator=NullGenerator(), N_points=1024, rng=rng):
    points = rng.normal(size=(N_points, 3))
    N_alg_objects = rng.integers(0, 8)
    alg_objects = [get_random_alg_object() for _ in range(N_alg_objects)]
    priority = rng.integers(-10, 10)
    tol = rng.uniform()
    return Volume(
        points=points,
        alg_objects=alg_objects,
        generator=generator,
        priority=priority,
        tolerance=tol,
    )


def perturb_volume(vol):
    new_vol = vol.from_volume()
    choice = rng.choice(["priority", "points", "alg_obj", "tolerance"])
    match choice:
        case "priority":
            while new_vol.priority == vol.priority:
                new_vol.priority = rng.integers(-10, 10)
        case "points":
            while array_set_equal(new_vol.points, vol.points):
                new_vol.points = rng.normal(size=(1024, 3))
        case "alg_obj":
            while EqualSet(new_vol.alg_objects) == EqualSet(vol.alg_objects):
                if len(vol.alg_objects) == 0:
                    new_vol.add_alg_object(get_random_alg_object())
                else:
                    new_vol._alg_objects = [
                        get_random_alg_object() for _ in range(len(vol.alg_objects))
                    ]
        case "tolerance":
            while np.isclose(new_vol.tolerance, vol.tolerance):
                new_vol.tolerance = rng.uniform()

    return new_vol


class Test_VolumeClass(czone_TestCase):
    def setUp(self):
        self.N_trials = 32

    def test_init(self):
        for _ in range(self.N_trials):
            vol = get_random_volume()
            self.assertReprEqual(vol)

    def test_eq(self):
        for _ in range(self.N_trials):
            vol = get_random_volume()

            new_vol = Volume(
                points=rng.permutation(vol.points, axis=0),
                alg_objects=list(rng.permutation(vol.alg_objects)),
                generator=NullGenerator(),
                priority=vol.priority,
                tolerance=vol.tolerance,
            )
            self.assertEqual(vol, new_vol)
            perturbed_volume = perturb_volume(vol)
            self.assertNotEqual(vol, perturbed_volume)

        new_vol = vol.from_volume(
            generator=Generator.from_spacegroup(
                [1], np.zeros((1, 3)), np.array([1, 1, 1]), np.array([90, 90, 90]), 225
            )
        )
        self.assertNotEqual(vol, new_vol)


class Test_MultiVolume(czone_TestCase):
    def setUp(self):
        self.N_trials = 32

    def test_init(self):
        for _ in range(self.N_trials):
            N_vols = rng.integers(1, 8)
            mvol = MultiVolume(
                [get_random_volume() for _ in range(N_vols)], priority=rng.integers(-10, 10)
            )
            self.assertReprEqual(mvol)

    def test_eq(self):
        for _ in range(self.N_trials):
            N_vols = rng.integers(1, 8)
            vols = [get_random_volume() for _ in range(N_vols)]
            priority = rng.integers(-10, 10)
            ref_mvol = MultiVolume(vols, priority=priority)
            test_mvol = MultiVolume(rng.permutation(vols), priority=priority)

            self.assertEqual(ref_mvol, test_mvol)
            self.assertNotEqual(ref_mvol, MultiVolume(vols, priority=priority + 1))

            perturbed_vols = [v for v in vols]
            idx = rng.choice(range(len(vols)))
            perturbed_vols[idx] = perturb_volume(perturbed_vols[idx])
            self.assertNotEqual(ref_mvol, MultiVolume(perturbed_vols, priority=priority))
