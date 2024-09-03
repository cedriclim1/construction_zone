from typing import List, Tuple

import numpy as np

from czone.transform import HStrain
from czone.volume import Plane


def round_away(x: float) -> float:
    """Round to float integer away from zero--opposite of np.fix.

    Args:
        x (float, ArrayLike[float]): number(s) to round

    Returns:
        float, ArrayLike[float]: rounded number(s)
    """
    return np.sign(x) * np.ceil(np.abs(x))


def get_N_splits(N: int, M: int, L: int, rng=None) -> List[int]:
    """Get N uniform random integers in interval [M,L-M) with separation at least M.

    Args:
        N (int): number of indices
        M (int): minimum distance between indices and ends of list
        L (int): length of initial list
        seed (int): seed for random number generator, default None

    Returns:
        List[int]: sorted list of random indices
    """
    if N == 0:
        return []

    if L - 2 * M < (N - 1) * M:
        raise ValueError(f"Minimum separation {M} is too large for {N} requested splits and length {L}")

    rng = np.random.default_rng() if rng is None else rng

    # seed an initial choice and create array to calculate distances in
    splits = [rng.integers(M, L - M)]
    data = np.array([x for x in range(M, L - M)])
    idx = np.ma.array(data=data, mask=np.abs(data - splits[-1]) < M)

    while len(splits) < N:
        while np.all(idx.mask):
            # no options left, reseed
            splits = [rng.integers(M, L - M)]
            idx.mask = np.abs(idx.data - splits[-1]) < M

        # add new choice to list and check distance against other indices
        splits.append(rng.choice(idx.compressed()))
        idx.mask = np.logical_or(idx.mask, np.abs(idx.data - splits[-1]) < M)

    splits.sort()
    return splits


def vangle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors of same dimension in R^N.

    Args:
        v1 (np.ndarray): N-D vector
        v2 (np.ndarray): N-D vector

    Returns:
        float: angle in radians
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def snap_plane_near_point(
    point: np.ndarray, generator, miller_indices: Tuple[int], mode: str = "nearest"
):
    """Determine nearest crystallographic nearest to point in space for given crystal coordinate system.

    Args:
        point (np.ndarray): Point in space.
        generator (Generator): Generator describing crystal coordinate system.
        miller_indices (Tuple[int]): miller indices of desired plane.
        mode (str): "nearest" for absolute closest plane to point; "floor" for
                    next nearest valid plane towards generator origin; "ceil"
                    for next furthest valid plane from generator origin.

    Returns:
        Plane in space with orientation given by Miller indices snapped to
        nearest valid location.

    """

    miller_indices = np.array(miller_indices)

    # check if generator has a strain field
    if generator.strain_field is None:
        # get point coordinates in generator coordinate system
        point_fcoord = np.array(np.linalg.solve(generator.voxel.sbases, point))
    else:
        assert isinstance(
            generator.strain_field, HStrain
        ), "Finding Miller planes with inhomogenous strain fields is not supported."

        if generator.strain_field.mode == "crystal":
            H = generator.strain_field.matrix
            point_fcoord = np.array(np.linalg.solve(H @ generator.voxel.sbases, point))

    # get lattice points that are intersected by miller plane
    with np.errstate(divide="ignore"):  # check for infs directly
        target_fcoord = 1 / miller_indices

    new_point = np.zeros((3, 1))

    # TODO: if bases are not orthonormal, this procedure is not correct
    # since the following rounds towards the nearest lattice points, with equal
    # weights given to all lattice vectors
    if mode == "nearest":
        for i in range(3):
            new_point[i, 0] = (
                np.round(point_fcoord[i] / target_fcoord[i]) * target_fcoord[i]
                if not np.isinf(target_fcoord[i])
                else point_fcoord[i]
            )
    elif mode == "ceil":
        for i in range(3):
            new_point[i, 0] = (
                round_away(point_fcoord[i] / target_fcoord[i]) * target_fcoord[i]
                if not np.isinf(target_fcoord[i])
                else point_fcoord[i]
            )
    elif mode == "floor":
        for i in range(3):
            new_point[i, 0] = (
                np.fix(point_fcoord[i] / target_fcoord[i]) * target_fcoord[i]
                if not np.isinf(target_fcoord[i])
                else point_fcoord[i]
            )

    if generator.strain_field is None:
        # scale back to real space
        new_point = generator.voxel.sbases @ new_point

        # get perpendicular vector
        normal = generator.voxel.reciprocal_bases.T @ miller_indices
    else:
        H = generator.voxel.sbases
        G = generator.strain_field.matrix
        new_point = G @ H @ new_point

        # get perpendicular vector
        normal = np.linalg.inv(H @ G).T @ miller_indices

    return Plane(normal=normal, point=new_point)
