import random
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset
import nrrd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from scrolls_instance_segmentation.data.cube_dataset_base import CubeLoader


class InstanceCubesDataset(IterableDataset):
    """
    Datamodule for real scroll cubes and the corresponding instance masks.

    Path to dataset requires a folder structure of:
    [parent_folder]
        - volumes
        - layers

    with the value to dataset_path being the path to the volumes folder eg:
    Path('/community-uploads/tim/dec_2024_submission_data/dec_2024_submission_training_set/volumes')

    spatial_transform applies a random flip/rotate/affine to train for orientation invariance
    layer_dropout removes some random layers for training data variety
    layer_shuffle randomly shuffles the order of the layer labels, as an instance task should be order invariant


    Cube size is hardcoded to match available training data of 256
    """


    def __init__(self,
                 dataset_path: Path,
                 spatial_transform: bool = True,
                 layer_dropout: bool = False,
                 layer_shuffle: bool = True,
                 ):
        self.cube_size = 256
        self.max_cube_size = 256
        self.spatial_transform = spatial_transform
        self.layer_dropout = layer_dropout
        self.layer_shuffle = layer_shuffle

        self.volume_list = list(dataset_path.glob('*_volume.nrrd'))
        self.cube_loader = CubeLoader()
        self.cube_path = dataset_path

    def __len__(self):
        return len(self.volume_list)

    def __iter__(self):
        random.shuffle(self.volume_list)

        for cube in self.volume_list:
            yield self._gather_batch(cube)

    def _get_label_and_volume(self, cube_path: Path):

        x_offset = np.random.randint(0, self.max_cube_size - self.cube_size + 1)
        y_offset = np.random.randint(0, self.max_cube_size - self.cube_size + 1)
        z_offset = np.random.randint(0, self.max_cube_size - self.cube_size + 1)

        vol, ___ = nrrd.read(str(cube_path))
        lbl, ___ = nrrd.read(str(cube_path).replace('volume', 'mask'))
        vol = vol.astype(np.float32, casting='safe')
        lbl = lbl.astype(np.uint8)

        vol = vol[x_offset:self.cube_size+x_offset, y_offset:y_offset+self.cube_size, z_offset:z_offset+self.cube_size]
        lbl = lbl[x_offset:self.cube_size+x_offset, y_offset:y_offset+self.cube_size, z_offset:z_offset+self.cube_size]
        return vol, lbl

    def _gather_batch(self, cube_path: Path):
        vol, lbl = self._get_label_and_volume(cube_path)

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

def worker_init_fn(test: int):
    random.seed((torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(seed=((torch.utils.data.get_worker_info().seed) % (2**32 - 1)))


class CubesDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: Path, batch_size: int = 1, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.training_set = InstanceCubesDataset(dataset_path)
        self.validation_set = InstanceCubesDataset(dataset_path)
        self.test_set = None

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.training_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn)

    def val_dataloader(self):
        return DataLoader(self.validation_set, batch_size=self.batch_size, num_workers=2, worker_init_fn=worker_init_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn)