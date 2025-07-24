from pathlib import Path
from synthetic_pages.types.homogeneous_transform import HomogeneousTransform
from synthetic_pages.types.mesh import Mesh
from synthetic_pages.types.nrrd import Nrrd
import numpy as np
from synthetic_pages.utils import compute_signed_distances


def load_page_meshes_from_zyx_header(position_header: str, labels_dir: Path):
    """
    Position header is a string like "03840_02048_02560" which corresponds to ZYX coordinate.
    """
    mesh_files = labels_dir.glob(f'{position_header}_volume_mesh_*.obj')
    meshes_zyx = [Mesh.from_obj(labels_dir/fp) for fp in mesh_files]
    return meshes_zyx

def generate_label_volume(input_volume: Nrrd, page_meshes_zyx: list[Mesh],/,papyrus_threshold: int, page_thickness_unitless: float):
    """
    Generate instance labels based on proximity to page meshes. 
    """
    page_meshes_zyx =  [Mesh.from_trimesh(_m) for m in page_meshes_zyx for _m in m.as_trimesh(process=True).split(only_watertight=False)]

    def mask_as_pointcloud(
            mask, # (H, W, D)
    ):
        coordinates_zyx = np.argwhere(mask)
        coordinates_zyx_world = coordinates_zyx[:, None, :] @ input_volume.metadata['space directions'] + input_volume.metadata['space origin']
        return coordinates_zyx_world[:, 0, :]

    pcl = mask_as_pointcloud(input_volume.volume > papyrus_threshold)
    sdfs = np.array(
        [
            compute_signed_distances(
                pcl, 
                m, 
                distance_upper_bound = page_thickness_unitless/2) 
                for m 
                in page_meshes_zyx
        ]
    ).transpose((1,0))
    page_labels = np.argmin(sdfs, axis=-1) + 1 # page labels are positive indices
    page_labels[sdfs.min(axis=-1) > page_thickness_unitless / 2] = 0 # no page is the zero index
    mask = np.zeros_like(input_volume.volume, dtype=np.uint8)

    voxel_space_coordinates = (pcl - input_volume.metadata['space origin'])[:, None, :] @ input_volume.metadata['space directions'].T
    voxel_space_coordinates = voxel_space_coordinates.astype(np.uint32)
    mask[voxel_space_coordinates[:, 0, 0], voxel_space_coordinates[:, 0, 1], voxel_space_coordinates[:, 0 , 2]] = page_labels

    return Nrrd(mask, metadata=input_volume.metadata)
