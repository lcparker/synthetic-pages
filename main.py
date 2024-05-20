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

def deform_plane_as_grad(plane: Plane, bounding_box: BoundingBox2D, grid_density: int):
    spline2d = make_2d_spline(bounding_box)
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], grid_density)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], grid_density)

    x_points = spline2d[0](xs)
    y_points = spline2d[1](ys)

    X, Y = np.meshgrid(x_points, y_points)
    plane_grid = np.stack((X.T, Y.T, np.zeros(X.T), axis=-1)

    # transform the grid from implied coordinates to actual world coords
    # wait you don't need a plane you need a transform

    return plane_grid

class HomogeneousTransform:
                              def __init__(self, matrix=None):
        def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            assert matrix.shape == (4, 4), "Matrix must be of shape (4, 4)"
            self.matrix = matrix
    
# turn plane into mesh

# apply spline deformation to the mesh






# position mesh in space

# compute distances to mesh

"""
but is that really what we want? i don't think so

we want to define a 2d sheet and then get the set of distances from it to form a layer.



we should be able to do something like
* generate a plane mesh
* deform the plane mesh to be a spline
* for the grid compute distance to the surface
"""
