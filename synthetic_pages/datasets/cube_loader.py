from typing import Tuple
from abc import abstractmethod, ABC
from monai.transforms import Affine
import torch
import numpy as np

class CubeLoader(ABC):
    def one_hot(self, lbl: np.ndarray):
        return torch.nn.functional.one_hot(torch.from_numpy(lbl).unsqueeze(dim=0).long()).byte().permute(0, 4, 1, 2, 3)[0, :, :, :, :]

    @staticmethod
    def spatial_transform_logic(vol: np.ndarray, lbl: np.ndarray, cube_size: int):
        # TODO this is slow, can probably be made faster by transforming pages before voxelisation
        # flip vol randomly
        if np.random.randint(0, 2):
            lbl = np.flip(lbl, axis=0)
            vol = np.flip(vol, axis=0)
        if np.random.randint(0, 2):
            lbl = np.flip(lbl, axis=1)
            vol = np.flip(vol, axis=1)
        if np.random.randint(0, 2):
            lbl = np.flip(lbl, axis=2)
            vol = np.flip(vol, axis=2)

        # rotate vol randomly
        x_rotate = np.random.randint(0, 4)
        y_rotate = np.random.randint(0, 4)
        z_rotate = np.random.randint(0, 4)
        vol = np.rot90(vol, k=x_rotate, axes=(0, 1))
        vol = np.rot90(vol, k=y_rotate, axes=(0, 2))
        vol = np.rot90(vol, k=z_rotate, axes=(1, 2))
        lbl = np.rot90(lbl, k=x_rotate, axes=(0, 1))
        lbl = np.rot90(lbl, k=y_rotate, axes=(0, 2))
        lbl = np.rot90(lbl, k=z_rotate, axes=(1, 2))

        vol = vol.copy()
        lbl = lbl.copy()

        # affine vol randomly
        if np.random.random() > 0.35:
            vol = np.expand_dims(vol, 0)  # make them channel first
            lbl = np.expand_dims(lbl, 0)  # make a channel for the segmentation

            affine_severity = 12  # max = 45
            affine = Affine(
                rotate_params=((np.random.randint(-affine_severity, affine_severity + 1) * (np.pi / 180)),
                               (np.random.randint(-affine_severity, affine_severity + 1) * (np.pi / 180)),
                               (np.random.randint(-affine_severity, affine_severity + 1) * (np.pi / 180))),
                scale_params=((np.random.randint(9, 12) / 10),
                              (np.random.randint(9, 12) / 10),
                              (np.random.randint(9, 12) / 10)),
                padding_mode="zeros",
            )

            vol, _ = affine(vol, (cube_size, cube_size, cube_size), mode="bilinear")
            lbl, _ = affine(lbl, (cube_size, cube_size, cube_size), mode="nearest")

            lbl = lbl.byte()
            lbl = torch.where(vol == 0.0, torch.full_like(lbl, 0), lbl)

            vol = vol.squeeze().numpy()
            lbl = lbl.squeeze().numpy()

        return vol, lbl

    @staticmethod
    def shuffle_layers(layers: np.ndarray, # (N, H, W, D)
                             ) -> np.ndarray: # (N, H, W, D)
        """
        Rearranges the layers of a 3D label array by shuffling their order, while ensuring
        that the air layer stays in index 0.
        """
        assert len(layers.shape) == 4, "Input tensor must be 4D (N, H, W, D)"
        shuffled_indexes = torch.cat((torch.tensor([0]), torch.randperm(layers.size(0) - 1) + 1))
        shuffled_layers = layers[shuffled_indexes, :, :, :]

        assert len(shuffled_layers.shape) == 4 and shuffled_layers.shape == layers.shape

        return shuffled_layers

    @staticmethod
    def remove_empty_labels(lbl: np.ndarray):
        uniques = np.unique(lbl)
        largest_label = lbl.max()
        if (largest_label + 1) > len(uniques):
            for i in range(1, len(uniques)):
                lbl[lbl == uniques[i]] = i
        return lbl

    @staticmethod
    def dropout_page_layers(vol: np.ndarray, lbl: np.ndarray):

        # Pick some labels to replace with air
        existing_labels = np.unique(lbl)
        # immediately exit if there is less than 8 labels to begin with:
        if len(existing_labels) < 8:
            return vol, lbl
        existing_labels = np.array([val for val in existing_labels if val != 0])
        np.random.shuffle(existing_labels)
        labels_to_zero = existing_labels[:np.random.randint(len(existing_labels) - 6)]

        # Set the removed labels to air, and re-label as zero
        air_mask = (lbl == 0)
        replacement_mask = np.isin(lbl, labels_to_zero)
        replacement = np.random.choice(vol[air_mask], size=replacement_mask.sum())
        vol[replacement_mask] = replacement
        lbl[replacement_mask] = 0

        # Re-label the labels to avoid empty ones
        num_remaining_labels = len(existing_labels) - len(labels_to_zero) + 1
        available_labels = [idx for idx in range(1, num_remaining_labels) if
                            idx in labels_to_zero or idx not in existing_labels]

        for existing_label in existing_labels:
            if (
                    existing_label != 0
                    and existing_label not in labels_to_zero
                    and existing_label >= num_remaining_labels
            ):
                new_label = available_labels.pop()
                mask = (lbl == existing_label)
                lbl[mask] = new_label

        return vol, lbl


def worker_init_fn(test: int):
    random.seed((torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(seed=((torch.utils.data.get_worker_info().seed) % (2**32 - 1)))
