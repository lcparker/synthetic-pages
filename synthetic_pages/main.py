"""
Generate synthetic pages using bezier surfaces

The pages should be non-overlapping. 
"""

from math import comb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation
from typing import TypeVar

Point2D = tuple[float, float]
Point3D = tuple[float, float, float]
Indices3D = tuple[int, int, int]

class BoundingBox2D:
    def __init__(self, min: Point2D, max: Point2D):
        self.min = min
        self.max = max

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

def bernstein(index: int, degree: int, t: np.ndarray) -> np.ndarray:
    return comb(degree, index) * np.power(t, index) * np.power(1 - t, degree - index)

def __test_bernstein_visual():
    xs = np.linspace(0,1,100)
    for i in range(5):
        ys = [bernstein(i, 4 ,x) for x in xs]
        plt.plot(xs, ys)

    plt.show()

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

import vtk
from synthetic_pages.mesh import Mesh

def triangulate_pointcloud(pointcloud: np.ndarray) -> Mesh:
    """
    Triangulates the XY projection of pointcloud
    """
    assert pointcloud.shape[-1] == 3
    from scipy.spatial import Delaunay
    triangles = Delaunay(pointcloud[..., :2].reshape(-1, 2)).simplices
    return Mesh(pointcloud, triangles)

def XY_plane_grid(bounding_box: BoundingBox2D, num_points_per_axis: int = 10) -> np.ndarray:
    """
    turn bounding box into z=0 grid of bounds uniformly spaced inside
    bounding_box, with grid_density points per axis
    """
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], num_points_per_axis)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], num_points_per_axis)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    plane_grid = np.stack((X.T, Y.T, Z.T), axis=-1)
    return plane_grid

def unit_plane_3d(num_points_per_axis):
    unit_bbox = BoundingBox2D((0,0), (1,1))
    return XY_plane_grid(unit_bbox, num_points_per_axis).reshape(-1, 3)

def bezier_surface(
        control_points, # (H,W,3)
        num_points_per_axis: int
        ) -> Mesh:
    assert control_points.shape[-1] == 3
    assert len(control_points.shape) == 3

    pc = unit_plane_3d(num_points_per_axis)
    pc_3d = bezier_3d(control_points, pc)
    mesh = triangulate_pointcloud(pc_3d)
    return mesh

def _test_bezier_visual(control_points, bbox: BoundingBox2D):
    mesh = bezier_surface(control_points, 10)
    _, ax = mesh.scene_with_mesh_in_it()
    ax.scatter(control_points[..., 0], control_points[..., 1], control_points[..., 2])
    plt.show()

def make_control_points_3d(nx: int, ny: int, bounding_box: BoundingBox2D):
    xs = np.linspace(bounding_box.min[0], bounding_box.max[0], nx)
    ys = np.linspace(bounding_box.min[1], bounding_box.max[1], ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    control_points_3d = np.stack((X.T, Y.T, Z.T), axis=-1)
    return control_points_3d


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
        grid: np.ndarray,  # (...,3)
        mesh: Mesh,
        distance_upper_bound: float = 1e16) -> np.ndarray:
    """
    Compute signed distances from the supplied grid of points to the mesh. This
    only calculates distance to the nearest vertex, it doesn't interpolate.
    Will return infinity for any points that aren't within the upper bound
    provided.
    """
    from scipy.spatial import KDTree
    tree = KDTree(mesh.points)
    distances, _ = tree.query(grid.reshape(-1, 3), distance_upper_bound=distance_upper_bound)
    sdf = distances.reshape(grid.shape[:-1])
    return sdf

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
        else:
            raise ValueError(f"Unsupported type for matrix multiplication: {type(x)}")

    @staticmethod
    def translation(x: float, y: float, z: float):
        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = np.array([x,y,z])
        return HomogeneousTransform(translation_matrix)

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

if __name__ == '__main__':
    run_transform_tests()

###############################

def _test_transform_page(control_points, bbox_3d: BoundingBox3D):
    mesh = bezier_surface(control_points, num_points_per_axis=10)
    tf = HomogeneousTransform.random_transform(bbox_3d)
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
    mesh = bezier_surface(control_points, 10)
    # centre mesh at [0,0,0]
    mesh_centre = mesh.points.mean(axis=0)
    mesh.points = mesh.points - mesh_centre
    tf = HomogeneousTransform.random_transform(bbox_3d)

    mesh = tf.apply(mesh)

    _, ax = mesh.scene_with_mesh_in_it()
    ax = plot_bounding_box(ax, bbox_3d)
    plt.show()

### MULTI PAGE

def tesselate_pages(control_points, bbox_3d: BoundingBox3D, num_pages: int, spacing: float) -> list[Mesh]:
    """
    Tesselate `num_pages` deformed planes evenly spaced along the z-axis,
    'spacing' units apart. Transforms them to the centre of the bounding box
    provided and provided a random rotation.

    """
    mesh = bezier_surface(control_points, num_points_per_axis = 50)
    mesh_centre = mesh.points.mean(axis=0)
    mesh = Mesh(mesh.points - mesh_centre, mesh.triangles)
    zs = np.linspace(-spacing * num_pages/2, spacing * num_pages/2, num_pages)
    meshes= [HomogeneousTransform.translation(0,0,z).apply(mesh) for z in zs]
    tf = HomogeneousTransform.random_transform(bbox_3d)
    meshes= [tf.apply(mesh) for mesh in meshes]
    return meshes

def page_meshes_to_volume(page_meshes: list[Mesh], grid_size: int, page_thickness: float, bbox_3d: BoundingBox3D):
    grid = make_grid(bbox_3d, grid_size)
    sdfs = np.array([compute_signed_distances(grid, m, distance_upper_bound = page_thickness/2) for m in page_meshes]).transpose((1,2,3,0)) # (H, W, D, N)
    page_labels = np.argmin(sdfs, axis=-1) + 1 # page labels are positive indices
    page_labels[sdfs.min(axis=-1) > page_thickness / 2] = 0 # no page is the zero index
    return page_labels



def plot_bounding_box_vtk(renderer, bbox: BoundingBox3D):
    """Plot a 3D bounding box in a VTK renderer."""
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

    for edge in edges:
        line = vtk.vtkLineSource()
        line.SetPoint1(edge[0])
        line.SetPoint2(edge[1])
        line.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(line.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color
        actor.GetProperty().SetLineWidth(2)

        renderer.AddActor(actor)
    return renderer


def plot_meshes(meshes: list[Mesh],bbox: BoundingBox3D):
    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)
    
    for mesh in meshes:
        polydata = mesh.to_polydata()
        
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Add the actor to the renderer
        renderer.AddActor(actor)
    
    # Set background color and render the scene
    renderer = plot_bounding_box_vtk(renderer, bbox)
    renderer.SetBackground(0.1, 0.2, 0.3)
    render_window.Render()
    render_window_interactor.Start()


"""
Registration
* how to get differentiable output to minimise?
  * maybe use jax or something and do it implicitly (build graph and backprop through)?
  * randomly subsample target mesh, randomly subsample transformed grid (recreate subsample/mesh each iteration)
* hungarian algorithm to find correspondences
* figure out how to interpret everything else differentiably

some papers on differentiable assignment (related to classifying instances)
- https://arxiv.org/pdf/2211.14448
- https://logicalreasoninggnn.github.io/papers/10.pdf
- https://arxiv.org/pdf/1906.06618
- https://arxiv.org/pdf/2111.00030
- https://github.com/sharathadavanne/hungarian-net

"""


"""
from skimage.transform import resize
tim_mask = nrrd.read('mask.nrrd')
tim_mask = np.pad(resize(tim_mask[0], (16, 16, 16)), 1)
# nb naive marching cubes on this creates a ginormous mesh
print(f"shape of tim_mask is {tim_mask.shape}")

tim_mesh = mask_to_mesh(tim_mask)
print(f"load mesh has {len(tim_mesh.triangles)} triangles and {len(tim_mesh.points)} points")
tim_mesh.show()
"""

def save_labelmap(labelmap: np.ndarray, filename: str) -> None:
    import nrrd
    if len(labelmap.shape) != 3: raise ValueError("Labelmap must have shape (H, W, D)")
    header = {
        'type': 'int',
        'dimension': 3,
        'sizes': labelmap.shape,
        'space': 'left-posterior-superior',
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'raw',
        'space origin': [0.0, 0.0, 0.0],
        'space directions': [[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]]
    }
    nrrd.write(filename, labelmap, header)

from synthetic_pages.nrrd_file import *

if __name__ == "__main__":
    # generate control points in unit grid
    bbox = BoundingBox2D((0,0), (1,1)) 
    volume_bbox = BoundingBox3D((0,0,0), (1,1,1))