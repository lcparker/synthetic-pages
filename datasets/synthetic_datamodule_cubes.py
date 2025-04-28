import random
from typing import Tuple
import numpy as np
from torch.utils.data import IterableDataset
import nrrd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter

from scrolls_instance_segmentation.data.cube_dataset_base import CubeLoader

from synthetic_pages.main import (
    BoundingBox2D, 
    BoundingBox3D, 
    make_control_points_3d, 
    unit_plane_3d, 
    HomogeneousTransform, 
    Mesh, 
    triangulate_pointcloud, 
    page_meshes_to_volume,
)
from synthetic_pages.bezier_volume_deformation import bezier_space_deformation


class SyntheticInstanceCubesDataset(IterableDataset):
    """
    Produces a voxel cube of stacked synthetic pages that vary progressively.

    Shape of pages is generated using bezier surfaces. Requires reference volume and labels to choose intensity values
    when generating the synthetic pages.

    eg:
    reference_volume_filename = '/community-uploads/tim/dec_2024_submission_data/pretraining_reference/reference_slices.nrrd'
    reference_label_filename = '/community-uploads/tim/dec_2024_submission_data/pretraining_reference/reference_labels.nrrd'


    spatial_transform applies a random flip/rotate/affine to train for orientation invariance
    layer_dropout removes some random layers for training data variety
    layer_shuffle randomly shuffles the order of the layer labels, as an instance task should be order invariant

    Cube size is hardcoded to match available training data of 256
    """


    def __init__(self,
                 reference_volume_filename: str,
                 reference_label_filename: str,
                 spatial_transform: bool = True,
                 layer_dropout: bool = False,
                 layer_shuffle: bool = True):
        self.cube_size = 256
        self.max_cube_size = 256
        self.spatial_transform = spatial_transform
        self.layer_dropout = layer_dropout
        self.layer_shuffle = layer_shuffle

        self.reference_vol, ___ = nrrd.read(reference_volume_filename, index_order='C')
        self.reference_lbl, ___ = nrrd.read(reference_label_filename, index_order='C')
        self.reference_lbl, ___ = nrrd.read(reference_label_filename, index_order='C')

        self.air_intensity_mask = (self.reference_lbl == 0)
        self.outer_intensity_mask = (self.reference_lbl == 1)
        self.inner_intensity_mask = (self.reference_lbl == 2)

        self.cube_loader = CubeLoader()


    def __len__(self):
        return 50

    def __iter__(self):
        for i in range(500):
            yield self._gather_batch()

    def _gather_batch(self):
        vol, lbl = self._generate_label_and_vol()

        if self.layer_dropout:
            vol, lbl = self.cube_loader.dropout_page_layers(vol, lbl)

        if self.spatial_transform:
            vol, lbl = self.cube_loader.spatial_transform_logic(vol, lbl, cube_size=self.cube_size)

        lbl = self.cube_loader.remove_empty_labels(lbl)

        lbl = self.cube_loader.one_hot(lbl)
        if self.layer_shuffle:
            lbl = self.cube_loader.shuffle_layers(lbl)

        batch = {'vol': torch.from_numpy(vol),
                 'lbl': lbl}
        return batch

    def _generate_label_and_vol(self) -> Tuple[np.ndarray, np.ndarray]: # ((H, W, D), (H, W, D))
        """First pass. Required a LOT of turning and fiddling"""

        num_control_points = np.random.randint(3, 8)
        x_control_points = num_control_points
        y_control_points = num_control_points
        z_control_points = 5
        num_pages = np.random.randint(7, 16)
        bbox = BoundingBox2D((0, 0), (1, 1))

        control_points_3d = make_control_points_3d(x_control_points, y_control_points, bbox)
        control_points_3d[..., 2] = 0.5 * np.random.rand(*control_points_3d.shape[:-1])

        inital_page = control_points_3d.copy()
        stacked_control_points = []
        page_adjustment = [1, 1, 1]
        for i in range(z_control_points):
            page_adjustment = np.array(page_adjustment) + [0, 0, 0.15 * i + np.random.randint(0, 50) / 100]
            stacked_control_points.append((inital_page * page_adjustment) + [0, 0, num_pages / 150 * i])

        control_points = np.stack(stacked_control_points, axis=2)
        control_points = control_points - [0, 0, 0.25]

        pc = unit_plane_3d(num_points_per_axis=70)
        planes = [HomogeneousTransform.translation(0, 0, z).apply(pc) for z in np.linspace(0, 1, num_pages)]
        deformed_planes = [bezier_space_deformation(control_points, plane) for plane in planes]

        meshes = [Mesh(dp, triangulate_pointcloud(pc).triangles) for dp in deformed_planes]

        page_thickness = 0.05
        if num_pages > 10:
            page_thickness = 0.04
        labels = page_meshes_to_volume(meshes, 64, page_thickness, BoundingBox3D((0.1, 0.1, 0.1), (0.9, 0.9, 0.9)))
        labels = np.repeat(np.repeat(np.repeat(labels, 4, axis=0), 4, axis=1), 4, axis=2)
        inner_labels = page_meshes_to_volume(meshes, 64, page_thickness / 2, BoundingBox3D((0.1, 0.1, 0.1), (0.9, 0.9, 0.9)))
        inner_labels = np.repeat(np.repeat(np.repeat(inner_labels, 4, axis=0), 4, axis=1), 4, axis=2)


        # now add intensity/texture to make an output volume ------------------------------------------------
        output_vol = np.zeros_like(labels, dtype=np.float32)
        replacement_mask = labels == 0
        replacement = np.random.choice(self.reference_vol[self.air_intensity_mask], size=replacement_mask.sum())
        output_vol[replacement_mask] = replacement

        replacement_mask = labels > 0
        replacement = np.random.choice(self.reference_vol[self.outer_intensity_mask], size=replacement_mask.sum())
        output_vol[replacement_mask] = replacement

        output_vol = gaussian_filter(output_vol, sigma=1)

        replacement_mask = inner_labels > 0
        replacement = np.random.choice(self.reference_vol[self.inner_intensity_mask], size=replacement_mask.sum())
        output_vol[replacement_mask] = replacement
        output_vol = output_vol.astype(np.float32)

        # add a random offset to each label
        values = np.unique(labels)
        for value in values[1:]:
            output_vol[labels == value] += np.random.randint(-4000, 4000)

        output_vol = gaussian_filter(output_vol, sigma=1)
        vol_min = np.min(output_vol)

        # resample to be expected data range, i.e ~20000-65535
        output_vol = ((65535 - vol_min) * (output_vol - vol_min) / np.ptp(output_vol)) + vol_min

        return output_vol, labels


def worker_init_fn(test: int):
    random.seed((torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(seed=((torch.utils.data.get_worker_info().seed) % (2**32 - 1)))


class CubesDataModule(pl.LightningDataModule):
    def __init__(self,
                 reference_volume_filename: str,
                 reference_label_filename: str,
                 batch_size: int = 1,
                 num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_set = SyntheticInstanceCubesDataset(reference_volume_filename=reference_volume_filename,
                                                          reference_label_filename=reference_label_filename)
        self.validation_set = SyntheticInstanceCubesDataset(reference_volume_filename=reference_volume_filename,
                                                            reference_label_filename=reference_label_filename)
        self.test_set = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn, prefetch_factor=None)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn)
