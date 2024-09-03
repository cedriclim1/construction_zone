from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Union

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.symbols import Symbols

from czone.util.voxel import Voxel

""""
Generators
"""

class BaseGenerator(ABC):
    """Base abstract class for Generator objects.

    Generator objects are additive components in Construction Zone. When designing
    nanostructures, Generators contain information about the arrangement of atoms
    in space and can supply atoms at least where they should exist.

    BaseGenerators are typically not created directly. Use the Generator class
    for crystalline systems, and the AmorphousGenerator class for non-crystalline
    systems.
    """

    @abstractmethod
    def supply_atoms(self, bbox: np.ndarray):
        """Given a bounding region, supply enough atoms to complete fill the region.

        Args:
            bbox (np.ndarray): Nx3 array defining vertices of convex region

        Returns:
            Coordinates and species of atoms that fill convex region.
            Returned as Nx3 and Nx1 arrays.
        """
        pass

    @abstractmethod
    def transform(self, transformation: BaseTransform):
        """Transform Generator object with transformation described by Transformation object.

        Args:
            transformation (BaseTransform): Transformation object from transforms module.
        """
        pass

    def from_generator(self, **kwargs):
        """Constructor for new Generators based on existing Generator object.

        Args:
            **kwargs: "transformation"=List[BaseTransformation] to apply a
                        series of transformations to the copied generator.

        """
        new_generator = copy.deepcopy(self)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_generator.transform(t)

        return new_generator


"""
Volumes
"""

class BaseVolume(ABC):
    """Base abstract class for Volume objects.

    Volume objects are subtractive components in Construction Zone. When designing
    nanostructures, Volumes contain information about where atoms should and
    should not be placed. Semantically, volumes can be thought of as singular
    objects in space.

    BaseVolumes are typically not created directly. Use the Volume class for
    generalized convex objects, and the MultiVolume class for unions of convex
    objects.

    Attributes:
        atoms (np.ndarray): Nx3 array of atom positions of atoms lying within volume.
        species (np.ndarray): Nx1 array of atomic numbers of atoms lying within volume.
        ase_atoms (Atoms): Collection of atoms in volume as ASE Atoms object.
        priority (int): Relative generation precedence of volume.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @property
    def atoms(self):
        """Array of atomic positions of atoms lying within volume."""
        return self._atoms

    @property
    def species(self):
        """Array of atomic numbers of atoms lying within volume."""
        return self._species

    @property
    def ase_atoms(self):
        """Collection of atoms in volume as ASE Atoms object."""
        return Atoms(symbols=self.species, positions=self.atoms)

    @property
    def priority(self):
        """Relative generation precedence of volume."""
        return self._priority

    @priority.setter
    def priority(self, priority):
        if not np.issubdtype(type(priority), np.integer):
            raise TypeError("Priority needs to be integer valued")

        self._priority = int(priority)

    @abstractmethod
    def transform(self, transformation):
        """Transform volume with given transformation.

        Args:
            transformation (BaseTransform): transformation to apply to volume.
        """
        pass

    @abstractmethod
    def populate_atoms(self):
        """Fill volume with atoms."""
        pass

    @abstractmethod
    def checkIfInterior(self, testPoints: np.ndarray):
        """Check points to see if they lie in interior of volume.

        Returns:
            Logical array indicating which points lie inside the volume.
        """
        pass

    def to_file(self, fname, **kwargs):
        """Write object to an output file, using ASE write utilities.

        Args:
            fname (str): output file name.
            **kwargs: any key word arguments otherwise accepted by ASE write.
        """
        ase_write(filename=fname, images=self.ase_atoms, **kwargs)

    @abstractmethod
    def from_volume(self, **kwargs):
        pass


class BaseAlgebraic(ABC):
    """Base class for algebraic surfaces.


    Attributes:
        params (Tuple): parameters describing algebraic object
        tol (float): numerical tolerance used to pad interiority checks.
                    Default is 1e-5.

    """

    def __init__(self, tol: float = 1e-10):
        self.tol = tol

    @abstractmethod
    def checkIfInterior(self, testPoints: np.ndarray):
        """Check if points lie on interior side of geometric surface.

        Args:
            testPoints (np.ndarray): Nx3 array of points to check.

        Returns:
            Nx1 logical array indicating whether or not point is on interior
            of surface.
        """
        pass

    @property
    @abstractmethod
    def params(self):
        pass

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, val):
        assert float(val) >= 0.0
        self._tol = float(val)

    def from_alg_object(self, **kwargs):
        """Constructor for new algebraic objects based on existing Algebraic object.

        Args:
            **kwargs: "transformation"=List[BaseTransformation] to apply a
                        series of transformations to the copied generator.
        """
        new_alg_object = copy.deepcopy(self)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_alg_object = t.applyTransformation_alg(new_alg_object)

        return new_alg_object

"""
Scenes
"""
class BaseScene(ABC):
    def __init__(self, domain, objects):
        self._objects = []
        self._checks = []
        self.domain = domain
        self.add_object(objects)

    @property
    def domain(self) -> Voxel:
        """Current domain of nanoscale scene."""
        return self._domain

    @domain.setter
    def domain(self, domain: Voxel):
        if isinstance(domain, Voxel):
            self._domain = domain
        else:
            raise TypeError

    @property
    def objects(self) -> List[BaseVolume]:
        """List of objects in current scene."""
        return self._objects

    def add_object(self, ob: Union[BaseVolume, List[BaseVolume]]):
        """Add an object to the scene.

        Args:
            ob (BaseVolume): object or lists of objects to add to scene.
        """
        if ob is not None:
            ## create a temporary list to handle either single input or iter of inputs
            new_objects = []
            try:
                new_objects.extend(ob)
            except TypeError:
                new_objects.append(ob)

            ## check each new object to see if it is a Volume
            type_check = reduce(
                lambda x, y: x and y, [isinstance(new_obj, BaseVolume) for new_obj in new_objects]
            )
            if type_check:
                self._objects.extend(new_objects)
            else:
                raise TypeError(
                    f"Object {ob} must inherit from BaseVolume, and is instead {type(ob)}."
                )

    @property
    def _checks(self):
        """List of logical arrays indicating inclusion of atoms in scene from each object."""
        return self.__checks

    @_checks.setter
    def _checks(self, val):
        self.__checks = val

    # TODO: any way to cache these? is it worth it?
    @property
    def all_atoms(self):
        """Positions of all atoms currently in the scene after evaluating conflict resolution."""
        return np.vstack([ob.atoms[self._checks[i], :] for i, ob in enumerate(self.objects)])

    @property
    def all_species(self):
        """Atomic numbers of all atoms currently in the scene after evaluating conflict resolution."""
        return np.hstack([ob.species[self._checks[i]] for i, ob in enumerate(self.objects)])

    def species_from_object(self, idx: int):
        """Grab all the atoms from contributing object at idx.

        Returns:
            Numpy array of all positions of atoms contributed by object at idx.
        """
        return self.objects[idx].atoms[self._checks[idx], :]

    def _get_priorities(self):
        """Grab priority levels of all objects in Scene to determine precedence relationship.

        Returns:
            List of relative priority levels and offsets. Relative priority levels
            and offsets are used to determine which objects whill be checked
            for the inclusion of atoms in the scene of the atoms contributed by
            another object.

        """
        # get all priority levels active first
        self.objects.sort(key=lambda ob: ob.priority)
        plevels = np.array([x.priority for x in self.objects])

        # get unique levels and create relative priority array
        __, idx = np.unique(plevels, return_index=True)
        rel_plevels = np.zeros(len(self.objects)).astype(int)
        for i in idx[1:]:
            rel_plevels[i:] += 1

        offsets = np.append(idx, len(self.objects))

        return rel_plevels, offsets

    @abstractmethod
    def check_against_object(self, atoms, idx):
        """Check to see if atoms are exterior to object at idx"""
        pass

    @abstractmethod
    def _prepare_for_population(self):
        pass

    def populate(self, check_collisions=True):
        """Populate the scene with atoms according to Volumes and priority levels.

        First, every object populates atoms against its own boundaries.
        Then, gather the list of priorities from all the objects.
        For each object, generate a True array of length ob.atoms.
        For each object in the same priority level or lower, perform interiority
        check and repeatedly perform logical_and to see if atoms belong in scene.

        - Lower priority numbers supercede objects with high priority numbers.
        - Objects on the same priority level will not supply atoms to the scene in their volume intersections.
        """

        self._prepare_for_population()
        for ob in self.objects:
            ob.populate_atoms()

        ## Sort objects by precedence and get packed list representation
        # offsets is array of length N_priority_levels + 1,
        # rel_plevels is array of length N_objects, where priorities are >= 0
        rel_plevels, offsets = self._get_priorities()

        self._checks = []

        ## TODO: add some heuristic checking for object collision,
        ## otherwise, with many objects, a lot of unneccesary checks
        for i, ob in enumerate(self.objects):
            check = np.ones(ob.atoms.shape[0]).astype(bool)

            if check_collisions:
                # Grab the final index of the object sharing current priority level
                eidx = offsets[rel_plevels[i] + 1]

                # Iterate over all objects up to priority level and check against their volumes
                for j in range(eidx):
                    if i != j:  # Required, since checking all objects with p_j <= p_i
                        check = np.logical_and(check, self.check_against_object(ob.atoms, j))

            self._checks.append(check)

    def populate_no_collisions(self):
        """Populate the scene without checking for object overlap. Use only if known by construction
        that objects have no intersection."""
        self.populate(check_collisions=False)

    def to_file(self, fname, **kwargs):
        """Write atomic scene to an output file, using ASE write utilities.

        If format="prismatic", will default to Debye-Waller factors of 0.1 RMS
        displacement in squared angstroms, unless dictionary of debye-waller factors
        is otherwise supplied.

        Args:
            fname (str): output file name.
            **kwargs: any key word arguments otherwise accepted by ASE write.
        """
        # TODO: refactor and allow for dwf to be specified
        if "format" in kwargs.keys():
            if kwargs["format"] == "prismatic":
                dwf = set(self.all_species)
                dw_default = (0.1**2.0) * 8 * np.pi**2.0
                dwf = {str(Symbols([x])): dw_default for x in dwf}
                ase_write(filename=fname, images=self.ase_atoms, debye_waller_factors=dwf, **kwargs)
        else:
            ase_write(filename=fname, images=self.ase_atoms, **kwargs)

"""
Transformations
"""
class BaseTransform(ABC):
    """Base class for transformation objects which manipulate Generators and Volumes.

    Transformation objects contain logic and parameters for manipulating the
    different types of objects used in Construction Zone, namely, Generators and
    Volumes. BaseTransform is typically not created directly. Use MatrixTransform for
    generalized matrix transformations.

    Attributes:
        locked (bool): whether or not transformation applies jointly to volumes containing generators
        basis_only (bool): whether or not transformation applies only to basis of generators
        params (tuple): parameters describing transformation

    """

    def __init__(self, locked: bool = True, basis_only: bool = False):
        self.locked = locked
        self.basis_only = basis_only

    @abstractmethod
    def applyTransformation(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation to a collection of points in space.

        Args:
            points (np.ndarray): Nx3 array of points to transform.

        Returns:
            np.ndarray: Nx3 array of transformed points.
        """
        pass

    @abstractmethod
    def applyTransformation_bases(self, points: np.ndarray) -> np.ndarray:
        """Apply transformation to bases of a generator.

        Args:
            points (np.ndarray): 3x3 array of bases from generator voxel.

        Returns:
            np.ndarray: transformed 3x3 array of bases.
        """
        pass

    @abstractmethod
    def applyTransformation_alg(self, alg_object: BaseAlgebraic) -> BaseAlgebraic:
        """Apply transformation to algebraic object.

        Args:
            alg_object (BaseAlgebraic): Algebraic object to transform.

        Returns:
            BaseAlgebraic: Transformed object.
        """
        pass

    @property
    @abstractmethod
    def params(self) -> tuple:
        """Return parameters describing transformation."""
        pass

    @property
    def locked(self) -> bool:
        """Boolean value indicating whether or not transformation jointly applied to Volumes and Generators."""
        return self._locked

    @locked.setter
    def locked(self, locked):
        assert isinstance(locked, bool), "Must supply bool to locked parameter."
        self._locked = locked

    @property
    def basis_only(self) -> bool:
        """Boolean value indicating whether or not transformation applied only to basis of Generators."""
        return self._basis_only

    @basis_only.setter
    def basis_only(self, basis_only):
        assert isinstance(basis_only, bool), "Must supply bool to basis_only parameter"
        self._basis_only = basis_only


class BasePostTransform(ABC):
    """Base class for post-generation pre-volume transformations."""

    def __init__(self):
        self.origin = np.array([0, 0, 0])

    @abstractmethod
    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):
        """Apply function to a collection of points and species

        Args:
            points (np.ndarray): Nx3 array of points in space
            species (np.ndarray): Nx1 array of corresponding species

        Returns:
            (np.ndarray, np.ndarray): Transformed arrays

        """
        pass


class BaseStrain(ABC):
    """Base class for strain fields that act on Generators.

    Strain objects can be attached to generators, and transform the coordinates
    of the atoms post-generation of the supercell. Strain fields apply strain
    in crystal coordinate system by default.

    Attributes:
        origin (np.ndarray): origin with which respect coordinates are strained
        mode (str): "crystal" or "standard", for straining in crystal coordinates
                    or for straining coordinates with respect to standard R3
                    orthonromal basis and orientation, respectively
        bases (np.ndarray): 3x3 array representing generator basis vectors
    """

    def __init__(self):
        self.origin = np.array([0, 0, 0])

    @abstractmethod
    def apply_strain(self, points: np.ndarray) -> np.ndarray:
        """Apply strain to a collection of points.

        Args:
            points (np.ndarray): Nx3 array of points in space

        Returns:
            np.ndarray: Nx3 array of strained points in space
        """
        pass

    def scrape_params(self, obj: BaseGenerator):  # noqa
        """Helper method to grab origin and bases from host generator.

        Args:
            obj (BaseGenerator): generator to grab parameters from
        """
        if self.mode == "crystal":
            self._bases = np.copy(obj.voxel.sbases)

        if self.origin_type == "generator":
            self.origin = np.copy(obj.origin)

    @property
    def origin(self):
        """Origin with respect to which strain is applied."""
        return self._origin

    @origin.setter
    def origin(self, val):
        assert val.shape == (3,), "Origin must have shape (3,)"
        self._origin = np.array(val)

    @property
    def origin_type(self):
        return self._origin_type

    @origin_type.setter
    def origin_type(self, val):
        self._origin_type = val

    @property
    def mode(self):
        """Coordinate system for strain application, either 'crystal' or 'standard'."""
        return self._mode

    @mode.setter
    def mode(self, val):
        if val == "crystal" or "standard":
            self._mode = val
        else:
            raise ValueError("Mode must be either crystal or standard")

    @property
    def bases(self):
        """ "Basis vectors of crystal coordinate system."""
        return self._bases
    

"""
Prefabs
"""

class BasePrefab(ABC):
    """Base abstract class for Prefab objects.

    Prefab objects are objects and classes that can run predesigned algorithms
    for generating certain classes of regular objects, typically with
    sampleable features or properties. For example, planar defects in FCC
    systems are easily described in algorithmic form-- a series of {111} planes
    can be chosen to put a defect on.

    Prefab objects will generally take in at least a base Generator object defining
    the system of interest and potentially take in Volume objects. They will return
    Volumes, or, more likely, MultiVolume objects which contains the resultant
    structure defined by the prefab routine.
    """

    @abstractmethod
    def build_object(self) -> BaseVolume:
        """Construct and return a prefabicated structure."""
        pass

    @property
    def rng(self):
        """Random number generator associated with Prefab"""
        return self._rng

    @rng.setter
    def rng(self, new_rng: np.random.BitGenerator):
        if not isinstance(new_rng, np.random.Generator):
            raise TypeError("Must supply a valid Numpy Generator")

        self._rng = new_rng
