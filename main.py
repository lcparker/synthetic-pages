"""
The plan:

I want to create a program that generates a volume with N synthetic pages.
It should output the data and the ground truth labels.

The pages should be non-overlapping. 

They should be in 3D.


WHAT TO DO
* get bezier surface with no transform working to deform a single point
* visualise a point cloud with deformations applied
* make it a mesh and deform that
* turn it into a volume based on within distance of page and turn that into voxel array


How do you generate N structured pages? Should be able to do something with splines?

The question is
* how do you generate realistic crumples for pages st they're not all just the exact same?
  and don't intersect?

My thoughts
* generate N straight pages (planes) oriented in a random direction in the scene
* deform them using control points (bezier curves or something like FFD)
* apply random local distortions/variations as like a masked kernel 
(so there's a heatmap over the whole volume that you scale the kernel with,
identity most places except where there are local variations)
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import BSpline

Spline2D = tuple[BSpline, BSpline]

def make_spline():
    xs = np.linspace(0,1,10)
    control_points = np.random.rand(len(xs))* 4 + 5
    spline = make_interp_spline(xs, control_points) # cubic b-spline fit to control points
    return spline

def random_control_points(seq):
    return np.random.rand(len(seq)) * 20

Point2D = tuple[float, float]
class BoundingBox2D:
    def __init__(self, min: Point2D, max: Point2D):
        self.min = min
        self.max = max

# make 2d spline
def make_2d_spline(bounding_box: BoundingBox2D) -> Spline2D:
    xs = np.linspace(bounding_box.min[0],bounding_box.max[0],10)
    x_cpts = random_control_points(xs)
    x_spline = make_interp_spline(xs, x_cpts)

    ys = np.linspace(bounding_box.min[1],bounding_box.max[1],10)
    y_cpts = random_control_points(ys)
    y_spline = make_interp_spline(ys, y_cpts)


    raise Exception("This code is broken! Deformations need to happen along z axis, look deeper")
    return (x_spline, y_spline)

# create plane object
class Plane:
    origin = np.array([0,0,0])
    normal = np.array([0,0,1])

    def __init__(self, origin, normal):
        self.origin = origin
        self.normal = normal


class HomogeneousTransform:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            assert matrix.shape == (4, 4), "Matrix must be of shape (4, 4)"
            self.matrix = matrix

    def apply(self, points: np.ndarray) -> np.ndarray:
        # Ensure points are of shape (..., 3)
        assert points.shape[-1] == 3, "Points must have shape (..., 3)"
        
        # Convert points to homogeneous coordinates by adding a 1 in the last dimension
        homogeneous_coordinates = np.concatenate([points, np.ones(points.shape[:-1])[..., None]], axis=-1)
        transformed_points_homogeneous = homogeneous_coordinates @ self.matrix.T

        # Convert back to 3D coordinates
        transformed_points = transformed_points_homogeneous[..., :3] / transformed_points_homogeneous[..., 3, np.newaxis]
        
        return transformed_points


import unittest
from numpy.testing import assert_array_almost_equal

class TestHomogeneousTransform(unittest.TestCase):
    
    def test_identity_transformation(self):
        transform = HomogeneousTransform()
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        transformed_points = transform.apply(points)
        assert_array_almost_equal(transformed_points, points)

    def test_translation(self):
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = [1, 2, 3]
        transform = HomogeneousTransform(translation_matrix)
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        transformed_points = transform.apply(points)
        expected_points = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        assert_array_almost_equal(transformed_points, expected_points)

    def test_scaling(self):
        scaling_matrix = np.diag([2, 3, 4, 1])
        transform = HomogeneousTransform(scaling_matrix)
        points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        transformed_points = transform.apply(points)
        expected_points = np.array([[2, 3, 4], [4, 6, 8], [6, 9, 12]])
        assert_array_almost_equal(transformed_points, expected_points)

    def test_rotation(self):
        angle = np.pi / 2
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        transform = HomogeneousTransform(rotation_matrix)
        points = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        transformed_points = transform.apply(points)
        expected_points = np.array([[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]])
        assert_array_almost_equal(transformed_points, expected_points)

    def test_combined_transformation(self):
        # Combined translation and scaling
        matrix = np.eye(4)
        matrix[:3, :3] = np.diag([2, 2, 2])
        matrix[:3, 3] = [1, 1, 1]
        transform = HomogeneousTransform(matrix)
        points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        transformed_points = transform.apply(points)
        expected_points = np.array([[3, 3, 3], [5, 5, 5], [7, 7, 7]])
        assert_array_almost_equal(transformed_points, expected_points)

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHomogeneousTransform)
    unittest.TextTestRunner().run(suite)

def deform_plane_as_grid(world_transform: HomogeneousTransform, bounding_box: BoundingBox2D, grid_density: int) -> np.ndarray:
    spline2d = make_2d_spline(bounding_box)
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], grid_density)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], grid_density)

    x_points = spline2d[0](xs)
    y_points = spline2d[1](ys)

    X, Y = np.meshgrid(x_points, y_points)
    plane_grid_3d = np.stack((X.T, Y.T, np.zeros(X.T.shape)), axis=-1)

    plane_grid_world = world_transform.apply(plane_grid_3d)

    return plane_grid_world.reshape(-1, 3)

import matplotlib.pyplot as plt
def visualize_grid(grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = grid[:, 0]
    ys = grid[:, 1]
    zs = grid[:, 2]

    ax.scatter(xs, ys, zs, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
# compute distances to mesh

def bounding_box_to_plane(bbox: BoundingBox2D, world_transform: HomogeneousTransform) -> "Mesh":
    """
    Creates a Trimesh object representing the bounding box as a plane with world_transform applied to it.
    """
    return None

class BoundingBox3D:
    def __init__(self, min: tuple[float, float, float], max: tuple[float, float, float]):
        self.min = min
        self.max = max

def make_grid(bbox: BoundingBox3D, points_per_axis: int) -> np.ndarray:
    xs = np.linspace(bbox.min[0], bbox.max[0], points_per_axis)
    ys = np.linspace(bbox.min[1], bbox.max[1], points_per_axis)
    zs = np.linspace(bbox.min[2], bbox.max[2], points_per_axis)

    X,Y,Z = np.meshgrid(xs, ys, zs)

    grid = np.stack((X.T,Y.T,Z.T), axis=-1)
    return grid

# NOW we want to compute the distance from the spline to the grid to get the
def distance_to_sheet(grid: np.ndarray, sheet: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


"""
How to generalise to multiple planes such that they don't intersect?
* make spline global
* then apply local deformations as a kernel if necessary
"""
