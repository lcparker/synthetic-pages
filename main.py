"""
The plan:

I want to create a program that generates a volume with N synthetic pages.
It should output the data and the ground truth labels.

The pages should be non-overlapping. 

They should be in 3D.


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
    return np.random.rand(len(seq))


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
        homogeneous_coordinates = np.concatenate([points, np.ones(points.shape[:-1])], axis=-1)
        transformed_points_homogeneous = homogeneous_coordinates @ self.matrix.T

        # Convert back to 3D coordinates
        transformed_points = transformed_points_homogeneous[..., :3] / transformed_points_homogeneous[..., 3, np.newaxis]
        
        return transformed_points

def deform_plane_as_grad(world_transform: HomogeneousTransform, bounding_box: BoundingBox2D, grid_density: int) -> np.ndarray:
    spline2d = make_2d_spline(bounding_box)
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], grid_density)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], grid_density)

    x_points = spline2d[0](xs)
    y_points = spline2d[1](ys)

    X, Y = np.meshgrid(x_points, y_points)
    plane_grid_3d = np.stack((X.T, Y.T, np.zeros(X.T)), axis=-1)

    plane_grid_world = world_transform.apply(plane_grid_3d)

    return plane_grid_world.reshape(-1, 3)
    
# compute distances to mesh

# NOW we want to compute the distance from the spline to the grid to get the

def bounding_box_to_plane(bbox: BoundingBox2D, world_transform: HomogeneousTransform) -> Mesh:
    """
    Creates a Trimesh object representing the bounding box as a plane with world_transform applied to it.
    """

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


def distance_to_sheet(grid: np.ndarray, sheet: np.ndarray) -> np.ndarray:
    raise NotImplementedError()


"""
How to generalise to multiple planes such that they don't intersect?
* make spline global
* then apply local deformations as a kernel if necessary
"""
