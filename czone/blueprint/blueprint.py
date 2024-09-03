from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from pymatgen.core import Lattice, Structure

from czone.generator import AmorphousGenerator, Generator, NullGenerator
from czone.scene import PeriodicScene, Scene
from czone.types import (
    BaseAlgebraic,
    BaseGenerator,
    BasePostTransform,
    BaseScene,
    BaseStrain,
    BaseVolume,
)
from czone.util.voxel import Voxel
from czone.volume import (
    Cylinder,
    MultiVolume,
    Plane,
    Sphere,
    Volume,
)


@dataclass(kw_only=True)
class BaseNode(ABC, Mapping):
    children: list = field(default_factory=list)

    # Do some type enforcement
    # ala https://stackoverflow.com/a/58992994
    def __post_init__(self):
        for name, field_type in self.__annotations__.items():
            if field_type == np.ndarray:
                setattr(self, name, np.array(getattr(self, name)))

    @property
    @abstractmethod
    def is_leaf(self) -> bool:
        pass

    @property
    @abstractmethod
    def base_type(self):
        pass

    @property
    @abstractmethod
    def class_type(self):
        pass

    @abstractmethod
    def add_node(self, node):
        pass

    def __getitem__(self, key):
        return self.__dict__[key]

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        yield from self.__dict__


class BaseGeneratorNode(BaseNode):
    @property
    def base_type(self):
        return BaseGenerator

    @property
    def class_type(self):
        return BaseGenerator

    @property
    def is_leaf(self):
        return True

    def add_node(self, node):
        pass


class NullGeneratorNode(BaseGeneratorNode):
    @property
    def class_type(self):
        return NullGenerator


@dataclass
class AmorphousGeneratorNode(BaseGeneratorNode):
    origin: np.ndarray
    min_dist: float
    density: float
    species: int

    @property
    def class_type(self):
        return AmorphousGenerator


@dataclass
class GeneratorNode(BaseGeneratorNode):
    origin: np.ndarray
    lattice_matrix: np.ndarray
    basis_species: tuple[int]
    basis_coords: np.ndarray
    strain_field: BaseStrain | None = None  # TODO: move to children?
    post_transform: BasePostTransform | None = None

    @property
    def class_type(self):
        return Generator


@dataclass
class BaseAlgebraicNode(BaseNode):
    tol: float

    @property
    def is_leaf(self):
        return True

    def add_node(self, node):
        pass

    @property
    def base_type(self):
        return BaseAlgebraic


@dataclass
class SphereNode(BaseAlgebraicNode):
    radius: float
    center: np.ndarray

    @property
    def class_type(self):
        return Sphere


@dataclass
class PlaneNode(BaseAlgebraicNode):
    normal: np.ndarray
    point: np.ndarray

    @property
    def class_type(self):
        return Plane


@dataclass
class CylinderNode(BaseAlgebraicNode):
    axis: np.ndarray
    point: np.ndarray
    radius: float
    length: float

    @property
    def class_type(self):
        return Cylinder


@dataclass
class VoxelNode(BaseNode):
    bases: np.ndarray
    origin: np.ndarray
    scale: float

    @property
    def base_type(self):
        return Voxel

    @property
    def class_type(self):
        return Voxel

    @property
    def is_leaf(self):
        return True

    def add_node(self, node):
        pass


@dataclass
class BaseVolumeNode(BaseNode):
    priority: int

    @property
    def base_type(self):
        return BaseVolume

    @property
    def is_leaf(self):
        return False


@dataclass
class VolumeNode(BaseVolumeNode):
    tolerance: float
    points: np.ndarray

    def add_node(self, node):
        match node:
            case BaseGeneratorNode():
                ## check if there are any Generators in children
                is_first_generator = True
                for i, c in enumerate(self.children):
                    if isinstance(c, BaseGeneratorNode):
                        is_first_generator = False
                        break

                if is_first_generator:
                    self.children.append(node)
                else:
                    self.children[i] = node

            case BaseAlgebraicNode():
                self.children.append(node)
            case _:
                raise TypeError("VolumeNodes can only be parent to BaseGeneratorNodes")

    @property
    def class_type(self):
        return Volume


class MultiVolumeNode(BaseVolumeNode):
    def add_node(self, node):
        match node:
            case VolumeNode() | MultiVolumeNode():
                self.children.append(node)
            case _:
                raise TypeError(
                    "MultiVolumeNodes can only be parent to other MultiVolumes or Volumes"
                )

    @property
    def class_type(self):
        return MultiVolume


class BaseSceneNode(BaseNode):
    @property
    def base_type(self):
        return BaseScene

    @property
    def is_leaf(self):
        return False

    def add_node(self, node):
        match node:
            case VoxelNode():
                is_first_voxel = True
                for i, c in enumerate(self.children):
                    if isinstance(c, VoxelNode):
                        is_first_voxel = False
                        break

                if is_first_voxel:
                    self.children.append(node)
                else:
                    self.children[i] = node

            case BaseVolumeNode():
                self.children.append(node)
            case _:
                raise TypeError


class SceneNode(BaseSceneNode):
    @property
    def class_type(self):
        return Scene


@dataclass
class PeriodicSceneNode(BaseSceneNode):
    pbc: tuple[bool]

    @property
    def class_type(self):
        return PeriodicScene


# If this were to grow significantly, perhaps it'd be better
# to utilize some class property implementation (see: https://stackoverflow.com/a/76301341)
# so that we can pull the class type name automatically in a comprehension,
# e.g. {n.class_type.__name__ : n.class_type for n in [nodes]}

## Used in Serializers
BaseNodeMap = {
    "NullGenerator": NullGeneratorNode,
    "AmorphousGenerator": AmorphousGeneratorNode,
    "Generator": GeneratorNode,
    "Sphere": SphereNode,
    "Plane": PlaneNode,
    "Cylinder": CylinderNode,
    "Voxel": VoxelNode,
    "Volume": VolumeNode,
    "MultiVolume": MultiVolumeNode,
    "Scene": SceneNode,
    "PeriodicScene": PeriodicSceneNode,
}

## Let class types be serialized as lowercase and uppercase
NodeMap = {
    **BaseNodeMap,
    **{k.lower(): v for k, v in BaseNodeMap.items()},
    **{k.upper(): v for k, v in BaseNodeMap.items()},
}


class Blueprint:
    """
    Represents (Periodic)Scenes, (Multi)Volumes, and Generators
    as nested mappings.
    """

    def __init__(self, obj):
        self.get_mapping(obj)

    def __repr__(self):
        return f"Blueprint({repr(self.to_object())})"

    def __eq__(self, other):
        if isinstance(other, Blueprint):
            other_obj = other.to_object()
        else:
            other_obj = other

        return self.to_object() == other_obj

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, node):
        if isinstance(node, BaseNode):
            self._mapping = node
        else:
            raise TypeError

    ####################
    # Forward mappings #
    ####################

    def get_mapping(self, obj):
        if isinstance(obj, BaseNode):
            self.mapping = obj
        else:
            self.mapping = self.forward_map(obj)

    @staticmethod
    def forward_map(obj) -> BaseNode:
        match obj:
            case BaseGenerator():
                return Blueprint.map_generator(obj)
            case BaseVolume():
                return Blueprint.map_volume(obj)
            case BaseScene():
                return Blueprint.map_scene(obj)
            case Voxel():
                return Blueprint.map_voxel(obj)
            case _:
                raise TypeError

    @staticmethod
    def map_generator(
        G: BaseGenerator,
    ) -> NullGeneratorNode | AmorphousGeneratorNode | GeneratorNode:
        match G:
            case NullGenerator():
                return NullGeneratorNode()
            case AmorphousGenerator():
                return AmorphousGeneratorNode(G.origin, G.min_dist, G.density, G.species)
            case Generator():
                params = {
                    "origin": G.origin,
                    "lattice_matrix": G.lattice.matrix,
                    "basis_species": G.species,
                    "basis_coords": G.coords,
                    "strain_field": G.strain_field,
                    "post_transform": G.post_transform,
                }
                return GeneratorNode(**params)
            case _:
                raise TypeError

    @staticmethod
    def map_algebraic(A: BaseAlgebraic) -> SphereNode | PlaneNode | CylinderNode:
        match A:
            case Sphere():
                return SphereNode(tol=A.tol, radius=A.radius, center=A.center)
            case Plane():
                return PlaneNode(tol=A.tol, point=A.point, normal=A.normal)
            case Cylinder():
                return CylinderNode(
                    tol=A.tol, radius=A.radius, length=A.length, axis=A.axis, point=A.point
                )
            case _:
                raise TypeError

    @staticmethod
    def map_volume(V: BaseVolume) -> VolumeNode | MultiVolumeNode:
        # Should be recursive, to handle multivolumes
        match V:
            case MultiVolume():
                node = MultiVolumeNode(V.priority)
                for o in V.volumes:
                    node.add_node(Blueprint.map_volume(o))
            case Volume():
                node = VolumeNode(priority=V.priority, tolerance=V.tolerance, points=V.points)
                node.add_node(Blueprint.map_generator(V.generator))
                for o in V.alg_objects:
                    node.add_node(Blueprint.map_algebraic(o))
            case _:
                raise TypeError

        return node

    @staticmethod
    def map_voxel(V: Voxel) -> VoxelNode:
        if isinstance(V, Voxel):
            return VoxelNode(V.bases, V.origin, V.scale)
        else:
            raise TypeError

    @staticmethod
    def map_scene(S: BaseScene) -> SceneNode | PeriodicSceneNode:
        match S:
            case Scene():
                node = SceneNode()
            case PeriodicScene():
                node = PeriodicSceneNode(pbc=S.pbc)
            case _:
                raise TypeError

        node.add_node(Blueprint.map_voxel(S.domain))
        for o in S.objects:
            node.add_node(Blueprint.map_volume(o))

        return node

    ####################
    # Inverse mappings #
    ####################

    def to_object(self):
        return self.inverse_map(self.mapping)

    @staticmethod
    def inverse_map(node: BaseNode):
        match node:
            case BaseGeneratorNode():
                return Blueprint.inverse_map_generator(node)
            case BaseAlgebraicNode():
                return Blueprint.inverse_map_algebraic(node)
            case BaseVolumeNode():
                return Blueprint.inverse_map_volume(node)
            case BaseSceneNode():
                return Blueprint.inverse_map_scene(node)
            case VoxelNode():
                return Blueprint.inverse_map_voxel(node)
            case _:
                raise TypeError

    @staticmethod
    def inverse_map_generator(
        node: BaseGeneratorNode,
    ) -> NullGenerator | Generator | AmorphousGenerator:
        params = {**node}
        children = params.pop("children")
        if len(children) != 0:
            raise ValueError("BaseGeneratorNodes should not have children.")

        match node:
            case NullGeneratorNode() | AmorphousGeneratorNode():
                return node.class_type(**params)
            case GeneratorNode():
                lattice = Lattice(params.pop("lattice_matrix"))
                species = params.pop("basis_species")
                coords = params.pop("basis_coords")
                structure = Structure(lattice, species, coords)
                return Generator(structure=structure, **params)
            case BaseGeneratorNode():
                raise TypeError("Base Generators should not be constructed directly.")
            case _:
                raise TypeError

    @staticmethod
    def inverse_map_algebraic(node: BaseAlgebraicNode) -> Sphere | Plane | Cylinder:
        params = {**node}
        children = params.pop("children")
        if len(children) != 0:
            raise ValueError("BaseAlgebraicNodes should not have children.")

        return node.class_type(**params)

    @staticmethod
    def inverse_map_volume(node: BaseVolumeNode) -> Volume | MultiVolume:
        params = {**node}
        children = params.pop("children")
        match node:
            case MultiVolumeNode():
                volumes = [Blueprint.inverse_map_volume(c) for c in children]
                return MultiVolume(volumes, **params)
            case VolumeNode():
                params["alg_objects"] = []
                for c in children:
                    if isinstance(c, BaseAlgebraicNode):
                        params["alg_objects"].append(Blueprint.inverse_map_algebraic(c))
                    else:
                        params["generator"] = Blueprint.inverse_map_generator(c)
                return Volume(**params)
            case BaseVolumeNode():
                raise TypeError("Base Volumes should not be constructed directly")
            case _:
                raise TypeError

    @staticmethod
    def inverse_map_voxel(node: VoxelNode) -> Voxel:
        params = {**node}
        children = params.pop("children")
        if len(children) != 0:
            raise ValueError("VoxelNodes should not have children.")

        return Voxel(**params)

    @staticmethod
    def inverse_map_scene(node: BaseSceneNode) -> Scene | PeriodicScene:
        if not isinstance(node, BaseSceneNode):
            raise TypeError

        params = {**node}
        children = params.pop("children")

        params["objects"] = []

        for c in children:
            match c:
                case VoxelNode():
                    params["domain"] = Blueprint.inverse_map_voxel(c)
                case BaseVolumeNode():
                    params["objects"].append(Blueprint.inverse_map_volume(c))
                case _:
                    raise TypeError

        return node.class_type(**params)
