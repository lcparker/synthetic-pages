"""
I want to create a program that generates a volume with N synthetic pages.
It should output the data and the ground truth labels.

The pages should be non-overlapping. 

They should be in 3D.

WHAT TO DO
* turn it into a volume based on within distance of page and turn that into voxel array

The question is
* how do you generate realistic crumples for pages st they're not all just the exact same?
  and don't intersect?

My thoughts
* generate N straight pages (planes) oriented in a random direction in the scene
* distort with control points
* apply random local distortions/variations as like a masked kernel 
(so there's a heatmap over the whole volume that you scale the kernel with,
identity most places except where there are local variations)
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

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

#################################

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

def run_transform_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHomogeneousTransform)
    unittest.TextTestRunner().run(suite)

run_transform_tests()


###############################

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


from math import comb
def bernstein(index: int, degree: int, t: np.ndarray) -> np.ndarray:
    return comb(degree, index) * np.power(t, index) * np.power(1 - t, degree - index)

def bezier(control_points: np.ndarray, p: np.ndarray) -> float:
    """
    Computes Q(u, v) = sum_over(i,j) B_i,j J_n,i(u) K_m,j(v)

    Where B_ij is the ijth control point (scalar) you can adjust the magnitude
    of the control point value, which is like moving it along the z axis.
    """ 
    assert p.shape[-1] == 2
    assert len(p.shape) == 3 # (H,W,2)
    n, m = control_points.shape
    u = p[..., 0]
    v = p[..., 1]
    
    B_u = np.array([bernstein(i, n - 1, u) for i in range(n)])
    B_v = np.array([bernstein(j, m - 1, v) for j in range(m)])
    
    B_uv = np.einsum('ijk, ljk -> iljk', B_u, B_v)
    zs = np.einsum('ij, ijkl -> kl',control_points, B_uv)
    return zs[..., None] # (H, W, 1)

def plane_pointcloud(bounding_box: BoundingBox2D, grid_density: int = 10):
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], grid_density)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], grid_density)
    X, Y = np.meshgrid(xs, ys)
    plane_grid = np.stack((X.T, Y.T), axis=-1)
    return plane_grid

def _test_bernstein_visual():
    xs = np.linspace(0,1,100)
    for i in range(5):
        ys = [bernstein(i, 4 ,x) for x in xs]
        plt.plot(xs, ys)

    plt.show()

def visualize_pointcloud(points: np.ndarray):
    assert points.shape[-1] == 3, "inner dimension of points array must be 3"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = points.reshape(-1, 3)[..., 0]
    ys = points.reshape(-1, 3)[..., 1]
    zs = points.reshape(-1, 3)[..., 2]

    ax.scatter(xs, ys, zs, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

from matplotlib.figure import Figure
from matplotlib.axes import Axes
class Mesh:
    def __init__(self, points, triangles):
        assert points.shape[-1] == 3
        assert len(points.shape) == 2
        assert triangles.shape[-1] == 3
        assert len(triangles.shape) == 2

        self.points = points
        self.triangles = triangles

    def show(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for triangle in self.triangles:
            closed_triangle = np.append(triangle, triangle[0])  # To close the triangle
            ax.plot(self.points[closed_triangle, 0], 
                    self.points[closed_triangle, 1], 
                    self.points[closed_triangle, 2], 'k-')

        ax.plot(self.points[..., 0], self.points[..., 1], self.points[..., 2], 'o', markersize=5, color='red')

        ax.set_title('3D Mesh with Delaunay Triangulation')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.show()

    def scene_with_mesh_in_it(self) -> tuple[Figure, Axes]:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for triangle in self.triangles:
            closed_triangle = np.append(triangle, triangle[0])  # To close the triangle
            ax.plot(self.points[closed_triangle, 0], 
                    self.points[closed_triangle, 1], 
                    self.points[closed_triangle, 2], 'k-')

        ax.plot(self.points[..., 0], self.points[..., 1], self.points[..., 2], 'o', markersize=5, color='red')

        ax.set_title('3D Mesh with Delaunay Triangulation')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        return fig, ax

def bezier_surface(control_points):
    n = control_points.shape[0] - 1
    m = control_points.shape[1] - 1
    bbox = BoundingBox2D((0,0), (1,1)) # generate unit grid
    pc = plane_pointcloud(bbox)
    z_coords = bezier(control_points, pc)
    coords_3d = np.concatenate((pc, z_coords), axis=-1).reshape(-1, 3)
    pc_3d = coords_3d.reshape(-1, 3)
    mesh = triangulate_points(pc_3d)
    return mesh


def control_points_3d(control_points, bounding_box: BoundingBox2D):
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], control_points.shape[0])
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], control_points.shape[1])

    X, Y = np.meshgrid(xs, ys)
    control_points_3d = np.stack((X.T, Y.T, control_points), axis=-1)
    return control_points_3d.reshape(-1, 3)

def _test_bezier_visual():
    n = 3
    m = 5
    control_points = np.random.rand(n+1,m+1)
    bbox = BoundingBox2D((0,0), (1,1))
    pc = plane_pointcloud(bbox)
    z_coords = bezier(control_points, pc)
    coords_3d = np.concatenate((pc, z_coords), axis=-1).reshape(-1, 3)
    pc_3d = coords_3d.reshape(-1, 3)
    mesh = triangulate_points(pc_3d)
    cp3d = control_points_3d(control_points, bbox)
    fig, ax = mesh.scene_with_mesh_in_it()
    ax.scatter(cp3d[..., 0], cp3d[..., 1], cp3d[..., 2])
    plt.show()
    # mesh.show()

    # visualize_pointcloud(coords_3d)

def triangulate_points(pointcloud: np.ndarray):
    assert pointcloud.shape[-1] == 3
    from scipy.spatial import Delaunay
    triangles = Delaunay(pointcloud[..., :2].reshape(-1, 2)).simplices
    return Mesh(pointcloud, triangles)

### MAKING THE VOLUME

def make_grid(bbox: BoundingBox3D, points_per_axis: int):
    xs = np.linspace(bbox.min[0], bbox.max[0], points_per_axis)
    ys = np.linspace(bbox.min[1], bbox.max[1], points_per_axis)
    zs = np.linspace(bbox.min[2], bbox.max[2], points_per_axis)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack((X, Y, Z), axis=-1)
    assert grid.shape[-1] == 3
    assert len(grid.shape) == 4
    return grid # (H,W,D,3)

def compute_signed_distances(
        grid: np.ndarray,  # (H,W,D,3)
        mesh: Mesh) -> np.ndarray:
    from scipy.spatial import KDTree
    tree = KDTree(mesh.points)
    distances, _ = tree.query(grid.reshape(-1, 3))
    sdf = distances.reshape(grid.shape[:-1])
    return sdf

from matplotlib.widgets import Slider
def visualize_mask_with_slider(mask):
    # Create the initial figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # Initial slice index
    initial_slice = mask.shape[0] // 2

    # Display initial slices
    slices = [mask[initial_slice, :, :],  # XY plane
              mask[:, initial_slice, :],  # XZ plane
              mask[:, :, initial_slice]]  # YZ plane

    images = []
    for i, ax in enumerate(axes):
        img = ax.imshow(slices[i], cmap='gray')
        ax.set_title(['XY Plane', 'XZ Plane', 'YZ Plane'][i])
        ax.axis('off')
        images.append(img)

    # Create axes for the slider
    axcolor = 'lightgoldenrodyellow'
    axslice = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor=axcolor)

    # Define the slider
    slice_slider = Slider(axslice, 'Slice', 0, mask.shape[0]-1, valinit=initial_slice, valfmt='%0.0f')

    # Update function to change the slice displayed
    def update(val):
        slice_idx = int(slice_slider.val)
        new_slices = [mask[slice_idx, :, :],  # XY plane
                      mask[:, slice_idx, :],  # XZ plane
                      mask[:, :, slice_idx]]  # YZ plane
        for img, new_slice in zip(images, new_slices):
            img.set_data(new_slice)
        fig.canvas.draw_idle()

    # Attach the update function to the slider
    slice_slider.on_changed(update)

    plt.show()

def create_mask(sdf, distance_threshold):
    mask = sdf <= distance_threshold
    return mask.astype(int)


import nibabel as nib
def save_mask_as_nifti(mask, filename):
    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(mask.astype(np.int16), affine=np.eye(4))
    
    # Save the image to file
    nib.save(nifti_img, filename)

def mask_to_mesh(mask) -> Mesh:
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(mask, level=0)
    return Mesh(verts, faces)

def plot_mesh(mesh: Mesh):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the individual vertices
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]

    # Create the triangular surface plot
    ax.plot_trisurf(x, y, mesh.triangles, z, cmap='viridis', lw=1, edgecolor='none')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Mesh with Filled Surface')

    plt.show()

n = 3
m = 5
control_points = np.random.rand(n+1,m+1)
bbox = BoundingBox2D((0,0), (1,1))
pc = plane_pointcloud(bbox, 100)
z_coords = bezier(control_points, pc)
coords_3d = np.concatenate((pc, z_coords), axis=-1).reshape(-1, 3)
pc_3d = coords_3d.reshape(-1, 3)
mesh = triangulate_points(pc_3d)

bbox3d = BoundingBox3D((-.5,-.5,-.5), (1.5,1.5,1.5))
grid = make_grid(bbox3d, 100)
sdf = compute_signed_distances(grid, mesh)
mask = create_mask(sdf, .05)
save_mask_as_nifti(mask, 'mask.nii')
mask_mesh = mask_to_mesh(mask)
plot_mesh(mask_mesh)
# visualize_mask_with_slider(mask)


"""
How to generalise to multiple planes such that they don't intersect?
* make spline global
* then apply local deformations as a kernel if necessary
"""
