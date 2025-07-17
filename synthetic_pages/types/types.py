import numpy as np

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Indices3D = tuple[int, int, int]

class Plane:
    normal: Point3D
    origin: Point3D

    def __init__(self, normal: Point3D, origin: Point3D):
        length = np.linalg.norm(normal)
        if not np.isclose(length, 1):
            raise ValueError(f"Unable to construct plane: direction vector has magnitude {length} instead of 1.")
        self.normal = normal
        self.origin = origin
