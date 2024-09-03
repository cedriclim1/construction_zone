from __future__ import annotations

from functools import reduce
from itertools import product

import numpy as np
from ase import Atoms

from czone.transform import Translation
from czone.types import BaseScene
from czone.util.eset import EqualSet
from czone.util.voxel import Voxel


class Scene(BaseScene):
    """Scene classes manage multiple objects interacting in space with cell boundaries.

    Attributes:
        bounds (np.ndarray): 2x3 array defining rectangular bounds of scene.
        objects (List[BaseVolume]): List of all objects currently in scene.
        all_atoms (np.ndarray): Coordinates of all atoms in scene after precedence checks.
        all_species (np.ndarray): Atomic numbers of all atoms in scene after precedence checks.
        ase_atoms (Atoms): Collection of atoms in scene as ASE Atoms object.
    """

    def __init__(self, domain: Voxel, objects=None):
        super().__init__(domain, objects)

    def __repr__(self) -> str:
        return f"Scene(domain={repr(self.domain)}, objects={repr(self.objects)})"

    def __eq__(self, other: Scene) -> bool:
        if isinstance(other, Scene):
            domain_check = self.domain == other.domain
            object_check = EqualSet(self.objects) == EqualSet(other.objects)
            return domain_check and object_check
        else:
            return False

    @property
    def ase_atoms(self):
        """Collection of atoms in scene as ASE Atoms object."""
        cell_dims = self.domain.sbases.T
        celldisp = self.domain.origin
        return Atoms(
            symbols=self.all_species, positions=self.all_atoms, cell=cell_dims, celldisp=celldisp
        )

    def check_against_object(self, atoms, idx):
        return np.logical_not(self.objects[idx].checkIfInterior(atoms))

    def _prepare_for_population(self):
        pass


class PeriodicScene(BaseScene):
    def __init__(self, domain: Voxel, objects=None, pbc=(True, True, True)):
        super().__init__(domain, objects)
        self.pbc = pbc

    def __repr__(self) -> str:
        return f"PeriodicScene(domain={repr(self.domain)}, objects={repr(self.objects)}, pbc={self.pbc})"

    def __eq__(self, other: PeriodicScene) -> bool:
        # TODO: a more expansive equality check should check on the folded periodic images of domain and pbc are equal
        if isinstance(other, PeriodicScene):
            domain_check = self.domain == other.domain
            pbc_check = self.pbc == other.pbc
            object_check = EqualSet(self.objects) == EqualSet(other.objects)
            return domain_check and object_check and pbc_check
        else:
            return False

    @property
    def pbc(self):
        return self._pbc

    @pbc.setter
    def pbc(self, val):
        if len(val) == 3:
            if reduce(lambda x, y: x and y, [np.issubdtype(type(v), bool) for v in val]):
                self._pbc = tuple(val)
            else:
                raise TypeError
        else:
            raise ValueError

    def _get_periodic_indices(self, bbox):
        """Get set of translation vectors, in units of the domain cell, for all
        relevant periodic images to generate."""

        cell_coords = self.domain.get_voxel_coords(bbox)

        pos_shifts = cell_coords < 0  # Volume needs to be shifted in positive directions
        neg_shifts = cell_coords >= 1  # Volume needs to be shifted in negative directions

        ps = [np.any(pos_shifts[:, i]) for i in range(cell_coords.shape[1])]
        ns = [np.any(neg_shifts[:, i]) for i in range(cell_coords.shape[1])]

        indices = [[0] for _ in range(cell_coords.shape[1])]
        for i, (p, n) in enumerate(zip(ps, ns)):
            if self.pbc[i]:
                if p and n:
                    raise AssertionError("Points extend through periodic domain")
                if p:
                    N_cells = -np.min(np.floor(cell_coords[:, i]))
                    indices[i] = [N_cells]
                    if (not np.all(pos_shifts[:, i])) or (N_cells > 1):
                        indices[i].append(N_cells - 1)
                if n:
                    N_cells = -np.max(np.floor(cell_coords[:, i]))
                    indices[i] = [N_cells]
                    if (not np.all(neg_shifts[:, i])) or (N_cells < -1):
                        indices[i].append(N_cells + 1)

        periodic_indices = set(product(*indices)).difference([(0, 0, 0)])
        return periodic_indices

    def _get_periodic_images(self):
        """Get periodic images of all objects."""
        self._periodic_images = {}
        for ob in self.objects:
            self._periodic_images[id(ob)] = []

            ## Determine which periodic images need to be generated
            bbox = ob.get_bounding_box()
            periodic_indices = self._get_periodic_indices(bbox)

            for pidx in periodic_indices:
                ## For each image, get a copy of volume translated to its periodic imnage
                pvec = np.array(pidx, dtype=int).reshape((3, -1))
                tvec = (self.domain.sbases @ pvec).reshape((3))
                transformation = [Translation(tvec)]
                new_vol = ob.from_volume(transformation=transformation)
                self._periodic_images[id(ob)].append(new_vol)

    def _get_folded_positions(self, points):
        domain_coords = self.domain.get_voxel_coords(points)

        fold_boundary = np.ones_like(domain_coords, dtype=bool)
        for i, p in enumerate(self.pbc):
            if not p:
                fold_boundary[:, i] = False

        folded_coords = np.mod(domain_coords, 1.0, out=domain_coords, where=fold_boundary)
        return self.domain.get_cartesian_coords(folded_coords)

    @property
    def periodic_images(self):
        return self._periodic_images

    def check_against_object(self, atoms, idx):
        pkey = id(self.objects[idx])
        return np.logical_not(
            reduce(
                lambda x, y: np.logical_or(x, y),
                [po.checkIfInterior(atoms) for po in self.periodic_images[pkey]],
                self.objects[idx].checkIfInterior(atoms),
            )
        )

    def _prepare_for_population(self):
        self._get_periodic_images()

    @property
    def all_atoms(self):
        return self._get_folded_positions(super().all_atoms)

    @property
    def ase_atoms(self):
        """Collection of atoms in scene as ASE Atoms object."""
        return Atoms(
            symbols=self.all_species,
            positions=self.all_atoms,
            cell=self.domain.sbases.T,
            pbc=self.pbc,
        )
