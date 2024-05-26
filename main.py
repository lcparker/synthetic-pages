"""
Generate synthetic pages using bezier surfaces

The pages should be non-overlapping. 

TODO
* how do you generate realistic crumples for pages st they're not all just the exact same?
  and don't intersect?
    -> i think this is easy using randomly initialised gaussian heatmaps and taking some kernel and conving * the intesnsity to get the modified image (so it's a smooth transform)
* work out how to get the sdf s.t. you can get N pages

LATER
* generate synthetic N-page blocks en masse for training

My thoughts
* generate N straight pages (planes) oriented in a random direction in the scene
* distort with control points
* apply random local distortions/variations as like a masked kernel 
* is it possible/feasible to generate the pages by hand manually adjusting the control points in 3d to deform the whole space, then getting the (u,v,w) coordinates of each point on the plane and deforming that
(so there's a heatmap over the whole volume that you scale the kernel with,
identity most places except where there are local variations)
"""

from math import comb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation

Point2D = tuple[float, float]
class BoundingBox2D:
    def __init__(self, min: Point2D, max: Point2D):
        self.min = min
        self.max = max


class BoundingBox3D:
    def __init__(self, min: tuple[float, float, float], max: tuple[float, float, float]):
        self.min = min
        self.max = max

def bernstein(index: int, degree: int, t: np.ndarray) -> np.ndarray:
    return comb(degree, index) * np.power(t, index) * np.power(1 - t, degree - index)

def bezier(control_points: np.ndarray, p: np.ndarray) -> np.ndarray:
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

def bezier_3d(control_points: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Computes Q(u, v) = sum_over(i,j,k) C_i,j,k B_n,i(u) B_m,j(v)

    Where C_ijk is the ijkth control point, a point in 3d space.
    """ 
    assert p.shape[-1] == 3
    assert len(p.shape) == 2 # (N,3)
    assert control_points.shape[-1] == 3
    assert len(control_points.shape) == 3
    n, m = control_points.shape[:-1]
    u = p[..., 0]
    v = p[..., 1]
    
    B_u = np.array([bernstein(i, n - 1, u) for i in range(n)]) # (n,N)
    B_v = np.array([bernstein(j, m - 1, v) for j in range(m)]) # (m,N)
    
    B_uvw = np.einsum('iN, jN -> Nij', B_u, B_v)
    pts = np.einsum('ija, Nij -> Na',control_points, B_uvw)
    return pts # (N, 3)

def plane_pointcloud_3d_coords(bounding_box: BoundingBox2D, grid_density: int = 10):
    """
    turn bounding box into z=0 grid of bounds uniformly spaced inside
    bounding_box, with grid_density points per axis
    """
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], grid_density)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], grid_density)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    plane_grid = np.stack((X.T, Y.T, Z.T), axis=-1)
    return plane_grid

def _test_bernstein_visual():
    xs = np.linspace(0,1,100)
    for i in range(5):
        ys = [bernstein(i, 4 ,x) for x in xs]
        plt.plot(xs, ys)

    plt.show()

class Mesh:
    def __init__(self, points, triangles):
        assert points.shape[-1] == 3
        assert len(points.shape) == 2
        assert triangles.shape[-1] == 3
        assert len(triangles.shape) == 2

        self.points = points
        self.triangles = triangles

    def show_wireframe(self):
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

    def scene_with_mesh_in_it(self, fig=None, ax=None) -> tuple[Figure, Axes]:
        fig = plt.figure(figsize=(10, 8)) if fig is None else fig
        fig = plt.figure(figsize=(10, 8)) if fig is None else fig
        ax = fig.add_subplot(111, projection='3d') if ax is None else ax

        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        ax.plot_trisurf(x, y, mesh.triangles, z, cmap='viridis', lw=1, edgecolor='none')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Mesh with Filled Surface')

        return fig, ax

    def show(self, fig=None, ax=None):
        fig = plt.figure(figsize=(10, 8)) if fig is None else fig
        ax = fig.add_subplot(111, projection='3d') if ax is None else ax

        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]

        ax.plot_trisurf(x, y, mesh.triangles, z, cmap='viridis', lw=1, edgecolor='none')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Mesh with Filled Surface')

        plt.show()

def make_control_points_3d(nx: int, ny: int, bounding_box: BoundingBox2D):
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], nx)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    control_points_3d = np.stack((X.T, Y.T, Z.T), axis=-1)
    return control_points_3d

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

def mesh_to_3d_page(mesh: Mesh, bbox: BoundingBox3D, distance=.05) -> np.ndarray:
    grid = make_grid(bbox3d, 100)
    sdf = compute_signed_distances(grid, mesh)
    mask = create_mask(sdf, .05)
    return mask

def unit_plane_3d():
    unit_bbox = BoundingBox2D((0,0), (1,1))
    return plane_pointcloud_3d_coords(unit_bbox).reshape(-1, 3)

def bezier_surface(
        control_points, # (H,W,3)
        ) -> Mesh:
    assert control_points.shape[-1] == 3
    assert len(control_points.shape) == 3

    pc = unit_plane_3d()
    pc_3d = bezier_3d(control_points, pc)
    mesh = triangulate_points(pc_3d)
    return mesh

def _test_bezier_visual(control_points, bbox: BoundingBox2D):
    mesh = bezier_surface(control_points, bbox)
    _, ax = mesh.scene_with_mesh_in_it()
    ax.scatter(control_points[..., 0], control_points[..., 1], control_points[..., 2])
    plt.show()

class HomogeneousTransform:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = np.eye(4)
        else:
            assert matrix.shape == (4, 4), "Matrix must be of shape (4, 4)"
            self.matrix = matrix

    def apply(self, x: Mesh | np.ndarray) -> Mesh | np.ndarray:
        if isinstance(x, Mesh):
            return self._apply_mesh(x)
        elif isinstance(x, np.ndarray):
            return self._apply_ndarray(x)
        else:
            raise ValueError("Unsupported type for matrix multiplication")

    def _apply_ndarray(self, points: np.ndarray) -> np.ndarray:
        # Ensure points are of shape (..., 3)
        assert points.shape[-1] == 3, "Points must have shape (..., 3)"
        
        # Convert points to homogeneous coordinates by adding a 1 in the last dimension
        homogeneous_coordinates = np.concatenate([points, np.ones(points.shape[:-1])[..., None]], axis=-1)
        transformed_points_homogeneous = homogeneous_coordinates @ self.matrix.T

        # Convert back to 3D coordinates
        transformed_points = transformed_points_homogeneous[..., :3] / transformed_points_homogeneous[..., 3, np.newaxis]
        
        return transformed_points

    def _apply_mesh(self, mesh: Mesh) -> Mesh:
        return Mesh(self.apply(mesh.points), mesh.triangles)


###### tests for homogeneous transforms ######

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
def generate_random_transform(bbox: BoundingBox3D) -> HomogeneousTransform:
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

def _test_transform_page(control_points, bbox: BoundingBox2D, bbox_3d: BoundingBox3D):
    mesh = bezier_surface(control_points, bbox)
    tf = generate_random_transform(bbox_3d)
    mesh = tf.apply(mesh)

    _, ax = mesh.scene_with_mesh_in_it()
    plt.show()

from mpl_toolkits.mplot3d.art3d import Line3DCollection
def plot_bounding_box(ax, bbox: BoundingBox3D):
    """Plot a 3D bounding box."""
    # Define the vertices of the bounding box
    vertices = np.array([
        [bbox.min[0], bbox.min[1], bbox.min[2]],
        [bbox.max[0], bbox.min[1], bbox.min[2]],
        [bbox.max[0], bbox.max[1], bbox.min[2]],
        [bbox.min[0], bbox.max[1], bbox.min[2]],
        [bbox.min[0], bbox.min[1], bbox.max[2]],
        [bbox.max[0], bbox.min[1], bbox.max[2]],
        [bbox.max[0], bbox.max[1], bbox.max[2]],
        [bbox.min[0], bbox.max[1], bbox.max[2]]
    ])
    
    # Define the edges of the bounding box
    edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[3]],
        [vertices[3], vertices[0]],
        [vertices[4], vertices[5]],
        [vertices[5], vertices[6]],
        [vertices[6], vertices[7]],
        [vertices[7], vertices[4]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[5]],
        [vertices[2], vertices[6]],
        [vertices[3], vertices[7]]
    ]
    
    # Create a 3D line collection from the edges
    edge_collection = Line3DCollection(edges, colors='r', linewidths=2)
    
    # Add the edge collection to the plot
    ax.add_collection3d(edge_collection)

    return ax

def _test_bounding_box_vis(control_points, bbox_3d: BoundingBox3D):
    mesh = bezier_surface(control_points)
    # centre mesh at [0,0,0]
    mesh_centre = mesh.points.mean(axis=0)
    mesh.points = mesh.points - mesh_centre
    tf = generate_random_transform(bbox_3d)

    mesh = tf.apply(mesh)

    _, ax = mesh.scene_with_mesh_in_it()
    ax = plot_bounding_box(ax, bbox_3d)
    plt.show()

def translation(x,y,z) -> HomogeneousTransform:
    # Generate random translation that keeps the object within the bounding box
    
    # Create translation matrix
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = np.array([x,y,z])
    return HomogeneousTransform(translation_matrix)

def tesselate_pages(control_points, bbox_3d, num_pages, spacing):
    mesh = bezier_surface(control_points)
    mesh_centre = mesh.points.mean(axis=0)
    mesh = Mesh(mesh.points - mesh_centre, mesh.triangles)
    zs = np.linspace(-spacing * num_pages/2, spacing * num_pages/2, num_pages)
    meshes= [translation(0,0,z).apply(mesh) for z in zs]
    tf = generate_random_transform(bbox_3d)
    meshes= [tf.apply(mesh) for mesh in meshes]

    fig, ax = meshes[0].scene_with_mesh_in_it()
    for m in meshes[1:]:
        fig, ax = m.scene_with_mesh_in_it(fig, ax)
    ax = plot_bounding_box(ax, bbox_3d)
    plt.show()

    return meshes

# mask = mesh_to_3d_page(mesh, bbox3d)

# save_mask_as_nifti(mask, 'mask.nii')
# mask_mesh = mask_to_mesh(mask)
# mask_mesh.show()

### EXAMPLE CODE ### 
n = 4
m = 6

# generate control points in unit grid
bbox = BoundingBox2D((0,0), (1,1)) 
volume_bbox = BoundingBox3D((0,0,0), (1,1,1))
control_points_3d = make_control_points_3d(n, m, bbox)

control_points_3d[..., 2] = np.random.rand(*control_points_3d.shape[:-1])

#_test_transform_page(control_points_3d, bbox, volume_bbox)
# _test_bounding_box_vis(control_points_3d, volume_bbox)
tesselate_pages(control_points_3d, volume_bbox, 9, .1)

"""
How to generalise to multiple planes such that they don't intersect?
* make spline global -> won't work
  * you have to only displace control points along the axis that you tesselate I THINK
* then apply local deformations as a kernel if necessary
"""
