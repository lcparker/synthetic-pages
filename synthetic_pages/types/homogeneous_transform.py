from typing import TypeVar
from synthetic_pages.types.bounding_box_3d import BoundingBox3D
from synthetic_pages.types.mesh import Mesh


import numpy as np
from scipy.spatial.transform import Rotation

from synthetic_pages.types.types import Point3D


Transformable = TypeVar('Transformable', Mesh, np.ndarray)
class HomogeneousTransform:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            assert matrix.shape == (4, 4), "Matrix must be of shape (4, 4)"
            self.matrix = matrix

    def apply(self, x: Transformable) -> Transformable:
        if isinstance(x, Mesh):
            return self._apply_mesh(x)
        elif isinstance(x, np.ndarray):
            return self._apply_ndarray(x)
        elif _is_Point3D(x):
            return self._apply_point(x)
        else:
            raise ValueError(f"Unsupported type for matrix multiplication: {type(x)}")

    @staticmethod
    def translation(point: Point3D):
        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = np.array(point)
        return HomogeneousTransform(translation_matrix)

    @staticmethod
    def flip(axis: Point3D):
        """
        Creates a flip transform along the provided axis. axis must be a normal vector
        """
        axis_length = np.linalg.norm(axis)
        if not np.isclose(axis_length-1., 0.):
            raise ValueError(f"Input axis must be a unit vector, but has length {axis_length}")

        vec = np.array(axis)[None, ...]
        matrix = np.eye(4)
  
        matrix[:3, :3] =  np.eye(3) - 2 * vec.T @ vec
        return HomogeneousTransform(matrix)


    @staticmethod
    def scale(scale_factor: int):
        if not _is_number(scale_factor):

            mat = np.eye(4)
            mat[:3, :3] *= scale_factor
            return HomogeneousTransform(mat)
        
    @staticmethod
    def random_rotation():
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = Rotation.random().as_matrix()
        return HomogeneousTransform(rotation_matrix)
        
    @staticmethod
    def random_transform(bbox: BoundingBox3D):
        # Generate random rotation
        rotation = Rotation.random().as_matrix()

        # Generate random scaling
        scale_factors = np.random.uniform(0.5, 1.5, size=3)
        scale_matrix = np.diag(np.append(scale_factors, 1))

        # Generate random translation that keeps the object within the bounding box
        # translation = np.random.uniform(bbox.min, bbox.max)

        # Create translation matrix
        translation_matrix = np.eye(4)
        # translation_matrix[:3, 3] = translation
        translation_matrix[:3, 3] = 0.5 * (np.array(bbox.min) + np.array(bbox.max))

        # Combine rotation, scaling, and translation into a single transformation matrix
        transform_matrix = translation_matrix @ scale_matrix
        transform_matrix[:3, :3] = transform_matrix[:3, :3] @ rotation

        return HomogeneousTransform(transform_matrix)

    def _apply_ndarray(self, points: np.ndarray) -> np.ndarray:
        # Ensure points are of shape (..., 3)
        assert points.shape[-1] == 3, "Points must have shape (..., 3)"

        # Convert points to homogeneous coordinates by adding a 1 in the last dimension
        homogeneous_coordinates = np.concatenate([points, np.ones(points.shape[:-1])[..., None]], axis=-1)
        transformed_points_homogeneous = homogeneous_coordinates @ self.matrix.T

        # Convert back to 3D coordinates
        transformed_points = transformed_points_homogeneous[..., :3] / transformed_points_homogeneous[..., 3, np.newaxis]

        return transformed_points

    def _apply_point(self, point: Point3D) -> np.ndarray:
        # Convert points to homogeneous coordinates by adding a 1 in the last dimension
        homogeneous_coordinates = np.array([point[0], point[1], point[2], 1])
        transformed_point_homogeneous = homogeneous_coordinates @ self.matrix.T

        # Convert back to 3D coordinates
        transformed_point = transformed_point_homogeneous[..., :3] / transformed_point_homogeneous[..., 3, np.newaxis]

        return transformed_point


    def _apply_mesh(self, mesh: Mesh) -> Mesh:
        return Mesh(self.apply(mesh.points), mesh.triangles)


def _is_Point3D(obj):
    return (isinstance(obj, tuple)
            and len(obj) == 3 
            and all(_is_number(x) for x in obj))


def _is_number(obj): return isinstance(obj, int|float)
