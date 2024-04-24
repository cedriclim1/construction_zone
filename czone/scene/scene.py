from typing import List, Union

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from ase.symbols import Symbols

from abc import ABC

from ..volume import BaseVolume, makeRectPrism
from functools import reduce


class BaseScene(ABC):

    def __init__(self):
        self._objects = []

    @property
    def objects(self):
        """List of objects in current scene."""
        return self._objects

    def add_object(self, ob: Union[BaseVolume, List[BaseVolume]]):
        """Add an object to the scene.
        
        Args:
            ob (BaseVolume): object or lists of objects to add to scene.
        """

        ## create a temporary list to handle either single input or iter of inputs
        new_objects = []
        try:
            new_objects.extend(ob)
        except TypeError:
            new_objects.append(ob)

        ## check each new object to see if it is a Volume
        type_check = reduce(lambda x, y: x and y,
                            [isinstance(new_obj, BaseVolume) for new_obj in new_objects])
        if type_check:
            self._objects.extend(new_objects)
        else:
            raise TypeError(f'Object {ob} must inherit from BaseVolume, and is instead {type(ob)}.')

    @property
    def _checks(self):
        """List of logical arrays indicating inclusion of atoms in scene from each object."""
        return self.__checks

    @_checks.setter
    def _checks(self, val):
        self.__checks = val

    #TODO: any way to cache these? is it worth it?
    @property
    def all_atoms(self):
        """Positions of all atoms currently in the scene after evaluating conflict resolution."""
        return np.vstack(
            [ob.atoms[self._checks[i], :] for i, ob in enumerate(self.objects)])

    @property
    def all_species(self):
        """Atomic numbers of all atoms currently in the scene after evaluating conflict resolution."""
        return np.hstack(
            [ob.species[self._checks[i]] for i, ob in enumerate(self.objects)])
    
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
                ase_write(filename=fname,
                          images=self.ase_atoms,
                          debye_waller_factors=dwf,
                          **kwargs)
        else:
            ase_write(filename=fname, images=self.ase_atoms, **kwargs)


class Scene(BaseScene):
    """Scene classes manage multiple objects interacting in space with cell boundaries.

    Attributes:
        bounds (np.ndarray): 2x3 array defining rectangular bounds of scene.
        objects (List[BaseVolume]): List of all objects currently in scene.
        all_atoms (np.ndarray): Coordinates of all atoms in scene after precedence checks.
        all_species (np.ndarray): Atomic numbers of all atoms in scene after precedence checks.
        ase_atoms (Atoms): Collection of atoms in scene as ASE Atoms object.
    """

    def __init__(self, bounds=None, objects=None):
        super().__init__()

        self._bounds = None
        self._checks = None

        if not (objects is None):
            self.add_object(objects)

        if bounds is None:
            #default bounding box is 10 angstrom cube
            bbox = makeRectPrism(10, 10, 10)
            self.bounds = np.vstack(
                [np.min(bbox, axis=0),
                 np.max(bbox, axis=0)])
        else:
            self.bounds = bounds

    @property
    def bounds(self):
        """Current boundaries of nanoscale scene."""
        return self._bounds

    @bounds.setter
    def bounds(self, bounds: np.ndarray):
        bounds = np.array(bounds)
        assert (bounds.shape == (2, 3))
        self._bounds = bounds

    @property
    def ase_atoms(self):
        """Collection of atoms in scene as ASE Atoms object."""
        cell_dims = self.bounds[1, :] - self.bounds[0, :]
        return Atoms(symbols=self.all_species,
                     positions=self.all_atoms,
                     cell=cell_dims)

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
                    if (i != j): # Required, since checking all objects with p_j <= p_i
                        check_against = np.logical_not(
                            self.objects[j].checkIfInterior(ob.atoms))
                        check = np.logical_and(check, check_against)

            self._checks.append(check)

    def populate_no_collisions(self):
        """Populate the scene without checking for object overlap. Use only if known by construction
            that objects have no intersection."""
        self.populate(check_collisions=False)


class PeriodicScene(BaseScene):
    pass