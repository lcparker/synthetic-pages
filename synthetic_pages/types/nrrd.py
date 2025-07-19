import nrrd
from pathlib import Path
from typing import Literal

import numpy as np
import torch

class Nrrd:
    def __init__(self, volume: np.ndarray, metadata) -> None:
        if len(volume.shape) != 3: raise ValueError("Nrrd volume must have shape (H, W, D)")
        self.__is_valid_metadata(metadata)
        if not np.all(np.array(volume.shape) == metadata['sizes']):
            raise ValueError("Can't create nrrd object: volume shape does not match metadata")
        self.volume = volume
        self.metadata = metadata

    def __eq__(self, other) -> bool:
        volumes_eq = bool(np.all(self.volume == other.volume))
        def entry_eq(k1, k2):
            if isinstance(k1, np.ndarray) or isinstance(k2, np.ndarray):
                return np.all(k1 == k2)
            else:
                return k1 == k2
        metatadata_eq = all(
                [
                k1 == k2
                and entry_eq(v1, v2)
                for ((k1, v1), (k2, v2))
                in zip(self.metadata.items(), other.metadata.items())
                ]
            )
        return volumes_eq and metatadata_eq

    def __is_valid_metadata(self, metadata) -> None:
        # TODO fill this out properly
        keys = metadata.keys()
        if not ('type' in keys
                or not 'dimension' in keys
                or not 'space' in keys 
                or not metadata['space'] == 'left-posterior-superior'
                or not 'sizes' in keys
                or not len(metadata['sizes']) == 3
                or not 'space directions' in keys
                or not isinstance(metadata['space directions'], np.ndarray)
                or not 'kinds' in keys
                or not 'encoding' in keys
                or not metadata['encoding'] in ['raw', 'gzip']
                or not 'space origin' in keys
                or not isinstance(metadata['space origin'], np.ndarray)
                or not len(metadata['space origin']) == 3):
            raise ValueError("Can't create nrrd object: invalid metadata")

    @staticmethod
    def from_file(file: str | Path, index_order: Literal["F", "C"] = "C"):
        volume, metadata = nrrd.read(str(file), index_order = index_order)
        return Nrrd(volume, metadata)

    @staticmethod
    def from_cube(cube, mask = True):
        """
        Imports a Cube object from the vesvius repo.

        No type hinting or direct import because importing the repo is slow.
        """
        volume = cube.mask if mask else cube.volume
        metadata = {
          'type': 'int64',
          'dimension': 3,
          'space': 'left-posterior-superior',
          'sizes': volume.shape,
          'space directions': np.array(
              [[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]]),
          'endian': 'little',
          'encoding': 'gzip',
          'space origin': np.array([cube.z, cube.y, cube.x]).astype(float)
                }
        return Nrrd(volume, metadata)

    @staticmethod
    def from_volume(volume: np.ndarray | torch.Tensor, metadata: dict|None = None):
        if isinstance(volume, torch.Tensor):
            volume = volume.numpy()  # Convert PyTorch tensor to NumPy array

        if len(volume.shape) != 3:
            raise ValueError("Volume must have shape (H, W, D)")

        # Use provided metadata or set reasonable defaults
        if metadata is None:
            metadata = {
                'type': str(volume.dtype),
                'dimension': 3,
                'space': 'left-posterior-superior',
                'sizes': volume.shape,
                'space directions': np.eye(3),
                'endian': 'little',
                'encoding': 'gzip',
                'space origin': np.zeros(3)
            }

        return Nrrd(volume, metadata)

    def write(self, filename: str | Path, index_order: Literal["F", "C"] = "C") -> None:
        nrrd.write(file = str(filename), data = self.volume, header = self.metadata, index_order=index_order)
