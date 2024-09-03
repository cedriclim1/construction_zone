import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .blueprint import (
    BaseNode,
    Blueprint,
    NodeMap,
)

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import tomlkit

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

try:
    import h5py

    H5PY_AVAIALBLE = True
except ImportError:
    H5PY_AVAIALBLE = False


class BaseSerializer(ABC):
    def __init__():
        pass

    @staticmethod
    @abstractmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        """Take a Blueprint and serialize to disk."""
        pass

    @classmethod
    def write(cls, filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        "Alias for serialize."
        cls.serialize(filepath, blueprint, **kwargs)

    @staticmethod
    @abstractmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        """Take a file and try to return a Blueprint"""
        pass

    @classmethod
    def read(cls, filepath: Path | str, **kwargs) -> Blueprint:
        """Alias for deserialize."""
        return cls.deserialize(filepath, **kwargs)


class json_Serializer(BaseSerializer):
    @staticmethod
    def to_dict(node: BaseNode) -> dict:
        res = {k: v for k, v in node.items() if v is not None}
        for k in res.keys():
            if isinstance(res[k], np.ndarray):
                res[k] = res[k].tolist()

        res["_class_type"] = node.class_type.__name__  # force to be first in sort order
        try:
            children = res.pop("children")
        except KeyError:
            children = []

        if len(children) > 0:
            res["children"] = [json_Serializer.to_dict(n) for n in children]

        return res

    @staticmethod
    def from_dict(bdict: dict) -> BaseNode:
        try:
            children = bdict.pop("children")
        except KeyError:
            children = []

        res = NodeMap[bdict.pop("_class_type")](**bdict)
        for n in children:
            res.add_node(json_Serializer.from_dict(n))

        return res

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        bdict = json_Serializer.to_dict(blueprint.mapping)

        try:
            indent = kwargs.pop("indent")
        except KeyError:
            indent = 4

        try:
            sort_keys = kwargs.pop("sort_keys")
        except KeyError:
            sort_keys = True

        with open(filepath, "w") as f:
            json.dump(bdict, f, sort_keys=sort_keys, indent=indent, **kwargs)

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        with open(filepath, "r") as f:
            bdict = json.load(f)

        node = json_Serializer.from_dict(bdict)
        return Blueprint(node)


class yaml_Serializer(BaseSerializer):
    # TODO: prettier formatting

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        bdict = json_Serializer.to_dict(blueprint.mapping)

        with open(filepath, "w") as f:
            yaml.dump(bdict, f, **kwargs)

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        with open(filepath, "r") as f:
            bdict = yaml.full_load(f)

        node = json_Serializer.from_dict(bdict)
        return Blueprint(node)


class toml_Serializer(BaseSerializer):
    # TODO: prettier formatting
    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        bdict = json_Serializer.to_dict(blueprint.mapping)

        with open(filepath, "w") as f:
            tomlkit.dump(bdict, f)

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        with open(filepath, "r") as f:
            bdict = tomlkit.load(f).unwrap()

        node = json_Serializer.from_dict(bdict)
        return Blueprint(node)


class h5_Serializer(BaseSerializer):
    # TODO: For now, adopting basic dictionary unfolding stategy, as in json, yaml, and toml
    # In future, for slightly more efficient packing, could adopt class-packing approach
    # e.g., if volume owns many planes can pack all plane params into one large array
    @staticmethod
    def write_node_to_group(node: BaseNode, group: h5py.Group, **kwargs) -> None:
        params = {**node}
        children = params.pop("children")

        group_name = kwargs.get("name", node.class_type.__name__)
        G = group.create_group(group_name)
        for k, v in params.items():
            match v:
                case None:
                    continue
                case np.ndarray():
                    dset = G.create_dataset(k, data=v)
                case _:
                    G.attrs[k] = v

        if len(children) > 0:
            # Get counters for children by type
            child_types = set([n.class_type for n in children])
            counters = {t: 0 for t in child_types}

            for n in children:
                t = n.class_type
                name = f"{t.__name__}_{counters[t]}"
                h5_Serializer.write_node_to_group(n, G, name=name)
                counters[t] += 1

    @staticmethod
    def read_node_from_group(group: h5py.Group) -> BaseNode:
        class_name = group.name.rsplit("/", 1)[-1].split("_")[0]

        params = dict(group.attrs)

        children = []
        for k in group.keys():
            if isinstance(group[k], h5py.Dataset):
                params[k] = np.array(group[k])
            else:
                children.append(k)

        node = NodeMap[class_name](**params)
        for cg in children:
            node.add_node(h5_Serializer.read_node_from_group(group[cg]))

        return node

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        head_node = blueprint.mapping
        with h5py.File(filepath, mode="w") as f:
            h5_Serializer.write_node_to_group(head_node, f)

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        with h5py.File(filepath, mode="r") as f:
            root_groups = list(f.keys())
            if len(root_groups) > 1:
                raise ValueError

            root_group = f[root_groups[0]]
            head_node = h5_Serializer.read_node_from_group(root_group)

        return Blueprint(head_node)


class Serializer(BaseSerializer):
    """Dispatch class."""

    @staticmethod
    def serialize(filepath: Path | str, blueprint: Blueprint, **kwargs) -> None:
        ## Get format from **kwargs if passed in; otherwise, infer from filepath
        output_format = kwargs.get("format", str(filepath).rsplit(".")[-1])

        match output_format:
            case "json":
                return json_Serializer.serialize(filepath, blueprint, **kwargs)
            case "h5" | "H5" | "hdf5":
                if H5PY_AVAIALBLE:
                    return h5_Serializer.serialize(filepath, blueprint, **kwargs)
                else:
                    raise ValueError(
                        "hdf5 support not available. Please install h5py: https://docs.h5py.org/"
                    )
            case "yaml":
                if YAML_AVAILABLE:
                    return yaml_Serializer.serialize(filepath, blueprint, **kwargs)
                else:
                    raise ValueError(
                        "yaml support not available. Please insall pyyaml: https://pyyaml.org"
                    )
            case "toml":
                if TOML_AVAILABLE:
                    return toml_Serializer.serialize(filepath, blueprint, **kwargs)
                else:
                    raise ValueError(
                        "toml support not available. Please insall tomlkit: https://tomlkit.readthedocs.io/en"
                    )
            case _:
                raise ValueError(f"Unsupported format {output_format} detected or passed in.")

    @staticmethod
    def deserialize(filepath: Path | str, **kwargs) -> Blueprint:
        ## Get format from **kwargs if passed in; otherwise, infer from filepath
        input_format = kwargs.get("format", str(filepath).rsplit(".")[-1])

        match input_format:
            case "json":
                return json_Serializer.deserialize(filepath, **kwargs)
            case "h5" | "H5" | "hdf5":
                if H5PY_AVAIALBLE:
                    return h5_Serializer.deserialize(filepath, **kwargs)
                else:
                    raise ValueError(
                        "hdf5 support not available. Please install h5py: https://docs.h5py.org/"
                    )
            case "yaml":
                if YAML_AVAILABLE:
                    return yaml_Serializer.deserialize(filepath, **kwargs)
                else:
                    raise ValueError(
                        "yaml support not available. Please insall pyyaml: https://pyyaml.org"
                    )
            case "toml":
                if TOML_AVAILABLE:
                    return toml_Serializer.deserialize(filepath, **kwargs)
                else:
                    raise ValueError(
                        "toml support not available. Please insall tomlkit: https://tomlkit.readthedocs.io/en"
                    )
            case _:
                raise ValueError(f"Unsupported format {input_format} detected or passed in.")
