from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from czone.types import BaseStrain

class HStrain(BaseStrain):
    """Strain class for applying homogeneous strain fields to generators.

    HStrain objects can be attached to generators, and transform the coordinates
    of the atoms post-generation of the supercell via simple strain tensor.
    HStrain fields apply strain in crystal coordinate system by default.

    Attributes:
        matrix (np.ndarray): Matrix representing homogeneous strain tensor.
                            Can be set with 3 (x,y,z), 6 (Voigt notation), or
                            9 values (as list or 3x3 array).
    """

    def __init__(self, matrix=None, origin="generator", mode="crystal"):
        super().__init__()
        if matrix is not None:
            self.matrix = matrix
        else:
            # apply no strain
            self._matrix = np.eye(3)

        self.mode = mode

        if origin != "generator":
            self.origin = origin
            self.origin_type = "global"
        else:
            self.origin_type = "generator"

        self._bases = None

    def __repr__(self):
        if self.origin_type == "generator":
            return f"HStrain(matrix={repr(self.matrix)}, origin='generator', mode='{self.mode}')"
        else:
            return f"HStrain(matrix={repr(self.matrix)}, origin={self.origin}, mode='{self.mode}')"

    def __eq__(self, other):
        if isinstance(other, HStrain):
            base_check = np.allclose(self.matrix, other.matrix) and self.mode == other.mode
            if self.origin_type == "generator":
                return base_check and self.origin_type == other.origin_type
            else:
                return base_check and self.origin == other.origin
        else:
            return False

    ##############
    # Properties #
    ##############

    @property
    def matrix(self):
        """Homogeneous strain tensor."""
        return self._matrix

    @matrix.setter
    def matrix(self, vals):
        vals = np.squeeze(np.array(vals))
        match vals.shape:
            case (3,):
                self._matrix = np.eye(3) * vals
            case (3, 3):
                self._matrix = vals
            case (9,):
                self._matrix = np.reshape(vals, (3, 3))
            case (6,):
                # voigt notation
                v = vals
                self._matrix = np.array(
                    [[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]]
                )
            case _:
                raise ValueError("Input shape must be either 3,6, or 9 elements")

    ##############
    ### Methods ##
    ##############
    def apply_strain(self, points: np.ndarray) -> np.ndarray:
        # get points relative to origin
        sp = np.copy(points) - self.origin

        if self.mode == "crystal":
            # project onto crystal coordinates, strain, project back into real space
            sp = sp @ np.linalg.inv(self.bases).T @ self.matrix @ self.bases.T
        else:
            # strain
            sp = sp @ self.matrix

        # shift back w.r.t. origin
        sp += self.origin

        return sp


class IStrain(BaseStrain):
    """Strain class for applying inhomogenous strain fields to generators.

    IStrain objects can be attached to generators, and transform the coordinates
    of the atoms post-generation of the supercell via arbitrary strain functions.
    IStrain fields apply strain in crystal coordinate system by default.

    User must input a custom strain function; strain functions by default should
    accept only points as positional arguments and can take any kwargs.

    Attributes:
        fun_kwargs (dict): kwargs to pass to custom strain function
        strain_fun (Callable): strain function F: R3 -> R3 for
                                np.arrays of shape (N,3)->(N,3)
    """

    def __init__(self, fun=None, origin="generator", mode="crystal", **kwargs):
        if fun is not None:
            self.strain_fun = fun
        else:
            # apply no strain
            self.strain_fun = lambda x: x

        self.mode = mode

        if origin != "generator":
            self.origin = origin
        else:
            super().__init__()

        self._bases = None
        self.fun_kwargs = kwargs

    ##############
    # Properties #
    ##############

    @property
    def fun_kwargs(self):
        """kwargs passed to custom strain function upon application of strain."""
        return self._fun_kwargs

    @fun_kwargs.setter
    def fun_kwargs(self, kwargs_dict: dict):
        assert isinstance(kwargs_dict, dict), "Must supply dictionary for arbirtrary extra kwargs"
        self._fun_kwargs = kwargs_dict

    @property
    def strain_fun(self):
        """Inhomogenous strain function to apply to coordinates."""
        return self._strain_fun

    @strain_fun.setter
    def strain_fun(self, fun: Callable[[np.ndarray], np.ndarray]):
        try:
            ref_arr = np.random.rand((100, 3))
            test_arr = fun(ref_arr, **self.fun_kwargs)
            assert test_arr.shape == (100, 3)
        except AssertionError:
            raise ValueError(
                "Strain function must return numpy arrays with shape (N,3) for input arrays of shape (N,3)"
            )

        self._strain_fun = copy.deepcopy(fun)

    ##############
    ### Methods ##
    ##############
    def apply_strain(self, points: np.ndarray) -> np.ndarray:
        # get points relative to origin
        sp = np.copy(points) - self.origin

        if self.mode == "crystal":
            # project onto crystal coordinates
            sp = sp @ np.linalg.inv(self.bases)

            # strain
            sp = self.strain_fun(sp, basis=self.bases, **self.fun_kwargs)

            # project back into real space
            sp = sp @ self.bases
        else:
            # strain
            sp = self.strain_fun(sp, **self.fun_kwargs)

        # shift back w.r.t. origin
        sp += self.origin

        return sp
