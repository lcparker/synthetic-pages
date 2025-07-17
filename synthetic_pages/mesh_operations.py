from typing import Optional
import numpy as np
from synthetic_pages.types.types import Plane, Point3D
from synthetic_pages.types.bounding_box_3d import BoundingBox
from synthetic_pages.types.mesh import Mesh

def clip_mesh_to_bounding_box(mesh: Mesh, bounding_box: BoundingBox) -> Mesh:
    bounding_planes = [
        Plane((1, 0, 0), (bounding_box.x_start, 0, 0)),
        Plane((-1, 0, 0), (bounding_box.x_end, 0, 0)),
        Plane((0, 1, 0), (0, bounding_box.y_start, 0)),
        Plane((0, -1, 0), (0, bounding_box.y_end, 0)),
        Plane((0, 0, 1), (0, 0, bounding_box.z_start)),
        Plane((0, 0, -1), (0, 0, bounding_box.z_end))
    ]

    mesh = mesh.as_trimesh().copy()
    for plane in bounding_planes:
        mesh = mesh.slice_plane(plane_origin=plane.origin, plane_normal=plane.normal)
    
    try:
        output_mesh = Mesh(mesh.vertices, mesh.faces)
        return output_mesh
    except ValueError as e:
        return None


def cut_mesh_with_plane(
    mesh: Mesh,
    plane: Plane,
) -> list[Point3D] | None:
    """
    Extract the 3D contour where a plane intersects a mesh.
    
    Args:
        mesh: The 3D mesh to section
        plane_origin: Point on the plane (3D coordinates)
        plane_normal: Normal vector of the plane (3D coordinates)
    
    Returns:
        Path3D object containing the intersection contour(s) in 3D space
    """
    # Get the 2D section first
    section_2d = mesh.as_trimesh().section(
        plane_origin=plane.origin,
        plane_normal=plane.normal
    )
    
    if section_2d is None:
        return None
    
    contour_points = np.array(section_2d.vertices)
    
    return contour_points