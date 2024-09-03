"""
Short module for arbitrary post-generation, pre-volume transformations

Useful for chemical modifciations, statistical defects, etc.
"""

from __future__ import annotations

from typing import List

import numpy as np

from czone.types import BasePostTransform

class ChemicalSubstitution(BasePostTransform):
    def __init__(self, mapping: dict, frac, rng=None):
        self.mapping = mapping
        self.frac = frac
        self.rng = np.random.default_rng() if rng is None else rng

    def __repr__(self):
        return f"ChemicalSubstitution(mapping={repr(self.mapping)}, frac={self.frac})"

    def __eq__(self, other):
        if isinstance(other, ChemicalSubstitution):
            return (self.mapping == other.mapping) and (np.isclose(self.frac, other.frac))
        else:
            return False

    @property
    def mapping(self) -> dict:
        return self._mapping

    @mapping.setter
    def mapping(self, m):
        if isinstance(m, dict):
            for k, v in m.items():
                if (not isinstance(k, int)) or (not isinstance(v, int)):
                    raise TypeError
                if k == v:
                    raise ValueError
            self._mapping = m
        else:
            raise TypeError

    @property
    def target(self):
        return self.mapping.keys()

    @property
    def substitute(self):
        return self.mapping.values()

    @property
    def frac(self) -> float:
        return self._frac

    @frac.setter
    def frac(self, val: float):
        if val <= 0 or val > 1:
            raise ValueError

        self._frac = val

    def _replace_species(self, species):
        out_species = np.copy(species)

        for t, s in zip(self.target, self.substitute):
            t_filter = species == t
            t_probs = self.rng.uniform(0, 1, size=species.shape)

            out_species[(t_filter) & (t_probs <= self.frac)] = s

        return out_species

    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):
        return points, self._replace_species(species, **kwargs)

    @property
    def rng(self):
        """Random number generator associated with Generator"""
        return self._rng

    @rng.setter
    def rng(self, new_rng: np.random.BitGenerator):
        if not isinstance(new_rng, np.random.Generator):
            raise TypeError("Must supply a valid Numpy Generator")

        self._rng = new_rng


class CustomPostTransform(BasePostTransform):
    def __init__(self, fun):
        self.fun = fun

    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):
        return self.fun(points, species)


class PostSequence(BasePostTransform):
    """Apply sequence of transforms"""

    def __init__(self, transforms: List[BasePostTransform]):
        self._transforms = []
        if not (transforms is None):
            self.add_transform(transforms)

    def add_transform(self, transform: BasePostTransform):
        """Add transform to Multitransform.

        Args:
            transform (Basetransform): transform object to add to Multitransform.
        """
        if hasattr(transform, "__iter__"):
            for v in transform:
                assert isinstance(v, BasePostTransform), "transforms must be transform objects"
            self._transforms.extend(transform)
        else:
            assert isinstance(transform, BasePostTransform), "transforms must be transform objects"
            self._transforms.append(transform)

    def apply_function(self, points: np.ndarray, species: np.ndarray, **kwargs):
        for t in self._transforms:
            points, species = t.apply_function(points, species, **kwargs)

        return points, species
