import numpy as np


class BoundingBox3D:
    def __init__(self, min: tuple[float, float, float], max: tuple[float, float, float]):
        self.min = min
        self.max = max

    def to_grid(self, nx: int, ny: int, nz: int) -> np.ndarray:
        """
        Returns (nx, ny, nz, 3) grid evenly spaced inside bounding box
        """
        xs = np.linspace(self.min[0], self.max[0], nx)
        ys = np.linspace(self.min[1], self.max[1], ny)
        zs = np.linspace(self.min[2], self.max[2], nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        control_points_3d = np.stack((X, Y, Z), axis=-1)
        return control_points_3d