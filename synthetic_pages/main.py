"""
Generate synthetic pages using bezier surfaces

The pages should be non-overlapping. 
"""

from math import comb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import TypeVar

from tests.test_homogeneous_transform import run_transform_tests
from synthetic_pages.homogeneous_transform import HomogeneousTransform
from synthetic_pages.types.bounding_box_2d import BoundingBox2D
from synthetic_pages.types.bounding_box_3d import BoundingBox3D

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
from synthetic_pages.types.mesh import Mesh

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