from typing import NamedTuple, Optional, Sequence
import numpy as np


class BoundingBox(NamedTuple):
    x_start: int | float
    x_end: int | float
    y_start: int | float
    y_end: int | float
    z_start: int | float
    z_end: int | float

    def to_grid(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """
        Returns (nx, ny, nz, 3) grid evenly spaced inside bounding box
        """
        xs = np.linspace(self.x_start, self.x_end, nx)
        ys = np.linspace(self.y_start, self.y_end, ny)
        zs = np.linspace(self.z_start, self.z_end, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        control_points_3d = np.stack((X, Y, Z), axis=-1)
        return control_points_3d

    @classmethod
    def from_min_max(cls, min_coords: Sequence[int | float], max_coords: Sequence[int | float]) -> 'BoundingBox':
        if len(min_coords) != 3 or len(max_coords) != 3:
            raise ValueError("Both min_coords and max_coords must have exactly 3 values")
        
        # Ensure min is actually less than max for each dimension
        for min_val, max_val, dim in zip(min_coords, max_coords, ['x', 'y', 'z']):
            if min_val >= max_val:
                raise ValueError(f"Min {dim} coordinate ({min_val}) must be less than max {dim} coordinate ({max_val})")
        
        return cls(
            x_start=min_coords[0],
            x_end=max_coords[0],
            y_start=min_coords[1],
            y_end=max_coords[1],
            z_start=min_coords[2],
            z_end=max_coords[2]
        )

    def validate(self, chunk_size: int = 256):
        """Ensure bounding box dimensions are valid for the given chunk size"""
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
            
        for dim, (start, end) in enumerate([
            (self.x_start, self.x_end),
            (self.y_start, self.y_end),
            (self.z_start, self.z_end)
        ], 1):
            if start < 0:
                raise ValueError(f"Dimension {dim} start coordinate ({start}) cannot be negative")
                
            if start % chunk_size != 0:
                raise ValueError(f"Dimension {dim} start coordinate ({start}) must be a multiple of chunk_size {chunk_size}")
                
            if (end - start) % chunk_size != 0:
                raise ValueError(f"Dimension {dim} size ({end-start}) must be a multiple of chunk_size {chunk_size}")

    @staticmethod
    def intersection(box1: 'BoundingBox', box2: 'BoundingBox') -> Optional['BoundingBox']:
        """Calculate the intersection of two bounding boxes. If no intersection exists, return None."""
        x_start = max(box1.x_start, box2.x_start)
        x_end = min(box1.x_end, box2.x_end)
        y_start = max(box1.y_start, box2.y_start)
        y_end = min(box1.y_end, box2.y_end)
        z_start = max(box1.z_start, box2.z_start)
        z_end = min(box1.z_end, box2.z_end)

        if x_start < x_end and y_start < y_end and z_start < z_end:
            return BoundingBox(x_start, x_end, y_start, y_end, z_start, z_end)
        return None