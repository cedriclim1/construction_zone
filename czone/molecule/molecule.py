from __future__ import annotations

import copy
import warnings

import numpy as np
from ase import Atoms
from pymatgen.core.structure import IMolecule

from czone.types import BaseGenerator, BaseTransform
from czone.util.eset import array_set_equal


class Molecule(BaseGenerator):
    """Base abstract class for Molecule objects.

    Molecule objects are intended to facilitate molecular atomic items, which
    are not easily generated in the Generation-Volume pair scheme. They are also
    intended to facilitate applications, for example, in surface chemistry studies.
    The molecule class mostly interfaces with other packages more suited for molecular
    generation.

    BaseMolecules are typically not created directly.

    Attributes:
        atoms (np.ndarray): Nx3 array of atom positions of atoms in molecule
        species (np.ndarray): Nx1 array of atomic numbers of atom in molecule
        origin (np.ndarray): Reference origin of molecule.
        orientation (np.ndarray): Reference orientation of molecule.
        ase_atoms (Atoms): Collection of atoms in molecule as ASE Atoms object

    """

    def __init__(self, species, positions, origin=None, **kwargs) -> None:
        self._atoms = None
        self._species = None
        self.reset_orientation()
        self.print_warnings = True
        self.set_atoms(species, positions)

        if origin is None:
            self.set_origin(point=np.array([0.0, 0.0, 0.0]))
        elif np.issubdtype(type(origin), np.integer):
            self.set_origin(idx=origin)
        else:
            self.set_origin(point=origin)

        if "orientation" in kwargs.keys():
            self.orientation = kwargs["orientation"]

    def __repr__(self) -> str:
        return f"Molecule(species={repr(self.species)}, positions={repr(self.atoms)})"

    def __eq__(self, other: Molecule) -> bool:
        if isinstance(other, Molecule):
            pos_check = array_set_equal(self.atoms, other.atoms)
            if pos_check:
                x_ind = np.argsort(self.atoms, axis=0)
                y_ind = np.argsort(other.atoms, axis=0)
                return np.array_equal(self.species[x_ind], other.species[y_ind])
        else:
            return False

    @property
    def print_warnings(self):
        return self._print_warnings

    @print_warnings.setter
    def print_warnings(self, val):
        if not isinstance(val, bool):
            raise TypeError

        self._print_warnings = val

    @property
    def atoms(self):
        """Array of atomic positions of atoms lying within molecule."""
        return self._atoms

    @property
    def species(self):
        """Array of atomic numbers of atoms lying within molecule."""
        return self._species

    def set_atoms(self, species, positions):
        # check size compatibilities; cast appropriately; set variables
        species = np.array(species)
        species = np.reshape(species, (-1,)).astype(int)
        positions = np.array(positions)
        positions = np.reshape(positions, (-1, 3))

        if positions.shape[0] != species.shape[0]:
            raise ValueError(
                f"Number of positions ({positions.shape[0]}) provided does not match number of species ({species.shape[0]}) provided"
            )

        self._species = species
        self._atoms = positions

    def update_positions(self, positions):
        positions = np.array(positions)
        positions = np.reshape(positions, self.atoms.shape)
        self._atoms = positions

    def update_species(self, species):
        species = np.array(species)
        species = np.reshape(species, self.species.shape).astype(int)
        self._species = species

    def remove_atoms(self, indices, new_origin_idx=None):
        """
        Args:
            indices: iterable(int), set of indices to remove
            new_origin_idx: int, original index number of atom to set as new origin
        """
        if new_origin_idx is not None:
            if not np.issubdtype(type(new_origin_idx), np.integer):
                raise TypeError("new_origin_idx must be an int")

            if np.abs(new_origin_idx) >= self.atoms.shape[0]:
                raise IndexError(
                    f"Supplied new_origin_idx {new_origin_idx} is out of bounds for {self.atoms.shape[0]} atom molecule"
                )

        if new_origin_idx in indices:
            raise IndexError(
                f"Supplied new_origin_idx {new_origin_idx} in set of indices of atoms to be removed."
            )

        if self._origin_tracking and self._origin_idx in indices:
            raise NotImplementedError  # TODO: Implement origin resetting behavior and warn user if origin is reset to a new index
            # self._origin_idx = new_origin_idx # TEST

        self._species = np.delete(self.species, indices, axis=0)
        self._atoms = np.delete(self.atoms, indices, axis=0)

    @property
    def ase_atoms(self):
        """Collection of atoms in molecule as ASE Atoms object."""
        return Atoms(symbols=self.species, positions=self.atoms)

    @property
    def origin(self):
        if self._origin_tracking:
            return self.atoms[self._origin_idx, :]
        else:
            return self._origin

    @property
    def _origin_tracking(self) -> bool:
        return self.__origin_tracking

    @_origin_tracking.setter
    def _origin_tracking(self, val: bool):
        assert isinstance(val, bool)

        self.__origin_tracking = val

    @property
    def _origin_idx(self) -> int:
        return self.__origin_idx

    @_origin_idx.setter
    def _origin_idx(self, val: int):
        if np.issubdtype(type(val), np.integer):
            if np.abs(val) < self.atoms.shape[0]:
                self.__origin_idx = val
            else:
                raise IndexError(
                    f"Supplied origin index is {val} is out of bounds for {self.atoms.shape[0]} atom molecule"
                )
        else:
            raise TypeError(f"Supplied drigin index is a {type(val)} and must be an integer")

    def transform(self, transformation: BaseTransform, transform_origin=True):
        """Transform molecule with given transformation.

        Args:
            transformation (BaseTransform): transformation to apply to molecule.
        """
        assert isinstance(
            transformation, BaseTransform
        ), "Supplied transformation not transformation object."

        self.set_atoms(self.species, transformation.applyTransformation(self.atoms))

        if transform_origin:
            if self._origin_tracking:
                if self.print_warnings:
                    warnings.warn(
                        f"Requested to transform molecule, but currently origin is set to track an atom. \n Origin will not be transformed. Molecule is currently tracking origin against atom {self._origin_idx}"
                    )
                return
            self.set_origin(point=transformation.applyTransformation(self.origin))

    def set_origin(self, point=None, idx=None) -> None:
        """Set the reference origin to global coordinate or to track specific atom.

        Args:
            point (np.ndarray):
            idx (int):
        """
        # TODO: switch to match statement in 3.10
        if point is not None:
            point = np.array(point).ravel()
            assert point.shape == (3,)
            self._origin_tracking = False
            self._origin = point

        elif idx is not None:
            self._origin_tracking = True
            self._origin_idx = idx

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, mat):
        # check for valid rotation matrix
        # rotation matrix transforms zone axes to global coordinate system
        if mat.shape != (3, 3):
            raise ValueError(f"Input matrix has shape {mat.shape}  but must have shape {(3,3)}.")

        if np.abs(np.linalg.det(mat) - 1.0) > 1e-6:
            raise ValueError("Input (rotation) matrix must have determinant of 1.")

        if np.sum(np.abs(mat @ mat.T - np.eye(3))) > 1e-6:
            raise ValueError(
                f"Input (rotation) matrix must be orthogonal."
            )  # TODO: provide info on non-orthogonal vectors

        self._orientation = mat

    def reset_orientation(self):
        """Reset orientation to align with global XYZ. Does not transform molecule."""
        self.orientation = np.eye(3)

    def supply_atoms(self, *args, **kwargs):
        return self.atoms, self.species

    # def checkIfInterior(self, testPoints: np.ndarray):
    #     ## TODO
    #     # have a minimum bond distance
    #     # perhaps set heuristically to maximum atomic radius for any of the constiuent atoms?
    #     warnings.warn("WARNING: Default behavior for interiority check for molecules not yet implemented. No atoms will be removed from Volume or Scene due to collisions with a higher priority Molecule.")
    #     return np.zeros(testPoints.shape[0], dtype=bool)

    @classmethod
    def from_ase_atoms(cls, atoms):
        if isinstance(atoms, Atoms):
            return cls(atoms.get_atomic_numbers(), atoms.get_positions())
        else:
            raise TypeError(f"Supplied atoms are {type(atoms)} and should be an ASE Atoms object")

    @classmethod
    def from_pmg_molecule(cls, atoms):
        if isinstance(atoms, IMolecule):
            species = [s.number for s in atoms.species]
            return cls(species, atoms.cart_coords)
        else:
            raise TypeError(
                f"Supplied atoms are {type(atoms)} and should be a Pymatgen IMolecule or Molecule object"
            )

    def from_molecule(self, **kwargs):
        """Constructor for new Molecules from existing Molecule object

        Args:
            **kwargs: "transformation"=List[BaseTransformation] to apply a
                        series of transformations to the copied molecule.
        """

        new_molecule = copy.deepcopy(self)

        if "transformation" in kwargs.keys():
            for t in kwargs["transformation"]:
                new_molecule.transform(t)

        return new_molecule
