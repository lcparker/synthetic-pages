from matplotlib import pyplot as plt
import numpy as np
from synthetic_pages.homogeneous_transform import HomogeneousTransform
from synthetic_pages.types.bounding_box_3d import BoundingBox3D
from synthetic_pages.main import bernstein, page_meshes_to_volume, save_labelmap, triangulate_pointcloud, unit_plane_3d
from synthetic_pages.types.mesh import Mesh


def bezier_space_deformation(control_points: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Deforms 3d points by warping the 3d space according to the defined control points.
    If you want the geometry to be well-defined, then the control points cannot cross over.

    Computes Q(u, v, w) = sum_over(i,j,k) C_i,j,k B_n,i(u) B_m,j(v) B_l,k(w)

    Where C_ijk is the ijkth control point, a point in 3d space. The indices (i,j,k)
    correspond to (x,y,z) axes, so for unit control points (a 3d evenly spaced grid) 
    control_points[:,0,0] should be a list of points whose x-coordinates increase as i increases while the y and z coordinates remain the same.
    """ 
    assert points.shape[-1] == 3
    assert len(points.shape) == 2 # (N,3)
    assert control_points.shape[-1] == 3
    assert len(control_points.shape) == 4
    n, m, l = control_points.shape[:-1]
    u = points[..., 0]
    v = points[..., 1]
    w = points[..., 2]
    
    B_u = np.array([bernstein(i, n - 1, u) for i in range(n)]) # (n,N)
    B_v = np.array([bernstein(j, m - 1, v) for j in range(m)]) # (m,N)
    B_w = np.array([bernstein(k, l - 1, w) for k in range(l)]) # (l,N)
    
    B_uvw = np.einsum('iN, jN, kN -> Nijk', B_u, B_v, B_w)
    pts = np.einsum('ijka, Nijk -> Na',control_points, B_uvw)
    return pts # (N, 3)

def plot_point_cloud(points: np.ndarray, fig = None, ax = None, color: str = 'b', marker: str = 'o'):
    """
    Plot a 3D point cloud using matplotlib.

    Args:
        points (np.ndarray): A (N, 3) array of 3D points.
        color (str): The color of the points (default: 'b' for blue).
        marker (str): The marker style (default: 'o' for circles).

    Example usage:
        points = np.random.rand(100, 3)
        plot_point_cloud(points)
    """
    assert points.shape[1] == 3, "Input points should be a (N, 3) array."

    fig = plt.figure(figsize=(10, 8)) if fig is None else fig
    ax = fig.add_subplot(111, projection='3d') if ax is None else ax
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, marker=marker)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    # plt.show()
    return fig, ax

def control_points_well_ordered(control_points: np.ndarray) -> bool:
    """
    Verify that the control points are well-ordered. 'Well-ordered' here means
    that none of the control points cross over eachother. This ensures that the
    bezier space deformation won't cause non-intersecting objects to intersect.
    """
    assert len(control_points.shape) == 4
    assert control_points.shape[-1] == 3

    for i in reversed(range(control_points.shape[0]-1)):
        if np.any(control_points[i+1,:,:,0] < control_points[i,:,:,0]):
            print(f'failed at x,{i}')
            return False
    for j in reversed(range(control_points.shape[1]-1)):
        if np.any(control_points[:,j+1,:,1] < control_points[:,j,:,1]):
            print(f'failed at y,{j}')
            return False
    for k in reversed(range(control_points.shape[2]-1)):
        if np.any(control_points[:,:,k+1,2] < control_points[:,:,k, 2]):
            print(f'failed at z,{k}')
            return False
    return True


def deform_control_points(control_points: np.ndarray) -> np.ndarray:
    """
    Apply a random shift to each control point, such that if the input is a
    uniform grid the grid will still be well-ordered.
    """
    assert len(control_points.shape) == 4
    assert control_points.shape[-1] == 3

    x_length = control_points[...,0].max() - control_points[...,0].min()
    y_length = control_points[...,1].max() - control_points[...,1].min()
    z_length = control_points[...,2].max() - control_points[...,2].min()

    adjustments = (np.random.random_sample(control_points.shape) - 0.5) * np.array(
        [
            x_length/(control_points.shape[0]-1), 
            y_length/(control_points.shape[1]-1), 
            z_length/(control_points.shape[2]-1), 
        ])
    return control_points + adjustments


if __name__ == "__main__":
    volume_bbox = BoundingBox3D((0,0,0), (1,1,1))
    control_points = volume_bbox.to_grid(5, 5, 5)


    control_points = deform_control_points(control_points)
    pc = unit_plane_3d(num_points_per_axis=40)
    planes = [HomogeneousTransform.translation(0,0,z).apply(pc) for z in np.linspace(0,1,10)]
    deformed_planes = [bezier_space_deformation(control_points, plane) for plane in planes]
    print(f'no overlaps?: {control_points_well_ordered(control_points)}')
    fig, ax = plot_point_cloud(control_points.reshape(-1, 3))
    plt.show()
    meshes = [Mesh(dp, triangulate_pointcloud(pc).triangles) for dp in deformed_planes]
    fig, ax = meshes[0].scene_with_mesh_in_it()
    for mesh in meshes[1:]:
        fig, ax = mesh.scene_with_mesh_in_it(fig=fig, ax=ax)

    plt.show()
    labels = page_meshes_to_volume(meshes, 128, .1, volume_bbox)
    save_labelmap(labels, 'labels.nrrd')
