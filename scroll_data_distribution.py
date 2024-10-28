hari_cubes = "hari-cubes/"

from pathlib import Path
from main import *
import numpy as np

cube_names = [p for p in Path(hari_cubes).iterdir() if not p.name.startswith(".")]


def load_volume_and_mask(directory: Path):
    files = list(directory.iterdir())
    volume = [f for f in files if str(f).find("volume") > 0][0]
    mask = [f for f in files if str(f).find("mask") > 0][0]
    return Nrrd.from_file(volume), Nrrd.from_file(mask)


from scipy.stats import gaussian_kde


def intensity_distribution_of_page(
    volume: Nrrd, mask: Nrrd, page_number: int
) -> gaussian_kde:
    page = volume.volume[mask.volume == page_number]
    distribution = gaussian_kde(page)
    return distribution


import pickle


def save_distribution(distribution: gaussian_kde, filename: str | Path):
    with open(filename, "wb") as f:
        pickle.dump(distribution, f)


def load_distribution(filename: str | Path) -> gaussian_kde:
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def make_distribution_files():
    paths = [p for p in Path(hari_cubes).iterdir() if not p.name.startswith(".")]
    for d in paths:
        print("processing directory", d)
        vol, mask = load_volume_and_mask(d)
        if not np.all(np.array(vol.volume.shape) == np.array([256, 256, 256])):
            print("skipping... wrong shape")
            continue
        print(
            f"loaded volume (shape {vol.volume.shape}) and mask (shape {mask.volume.shape})"
        )
        labels = np.unique(mask.volume)
        print("labels are ", labels)
        for page_no in labels:
            print(f"calculating intensity distribution {page_no}")
            kde = intensity_distribution_of_page(vol, mask, page_no)
            print("saving")
            save_distribution(kde, Path(d) / f"intensity_distribution_{page_no}.pkl")
        del vol, mask


"""
Secondary idea for intensity mapping: 
* take random windows of intensities and overlay them so they're overlapping, interpolating in the areas where they overlap.
* gaussian blur/conv blur


but try to get something training.
"""


def grep_volume_from_directory(directory: str | Path) -> Nrrd:
    files = [
        p
        for p in Path(directory).iterdir()
        if p.name.startswith("volume") and p.suffix == ".nrrd"
    ]
    if len(files) > 0:
        volume_file = files[0]
        volume = Nrrd.from_file(volume_file)
        return volume
    else:
        raise ValueError("Supplied folder does not contain a volume object")


def grep_labels_from_directory(directory: str | Path) -> Nrrd:
    files = [
        p
        for p in Path(directory).iterdir()
        if p.name.startswith("mask") and p.suffix == ".nrrd"
    ]
    if len(files) > 0:
        volume_file = files[0]
        volume = Nrrd.from_file(volume_file)
        return volume
    else:
        raise ValueError("Supplied folder does not contain a volume object")


def view_distributions_in_directory(directory: str | Path):
    xs = np.linspace(0, 100000, 1000)
    for p in [
        p for p in Path("hari-cubes/cube_0_2408_4560/").iterdir() if p.suffix == ".pkl"
    ]:
        kde = load_distribution(p)
        plt.plot(xs, kde(xs))
    plt.show()


def make_synthetic_page_volume(
    data_folder: str | Path = "hari-cubes/cube_0_2408_4560/",
    save_filename: str | None = None,
) -> tuple[Nrrd, Nrrd]:
    ground_truth_volume = grep_volume_from_directory(data_folder)
    if not len(set(ground_truth_volume.metadata["sizes"])) == 1:
        raise ValueError(
            f"Currently, only cubic volumes are supported, but size was {ground_truth_volume.metadata['sizes']}"
        )
    print(f"Creating page meshes for {data_folder}")
    labels = page_meshes_to_volume(
        meshes, ground_truth_volume.metadata["sizes"][0], 0.05, volume_bbox
    )
    synthetic_volume = labels.copy()
    classes = np.unique(labels)
    for f in Path(data_folder).iterdir():
        fstr = str(f)
        if fstr.endswith(".pkl"):
            kde = load_distribution(f)
            index = int(fstr.split(".")[0].split("_")[-1])
            if index in classes:
                mask = labels == index
                print(f"Applying synthetic texture to label {index}")
                synthetic_values = kde.resample(mask.sum()).flatten()
                synthetic_volume[mask] = synthetic_values

    synthetic_volume = Nrrd(synthetic_volume, ground_truth_volume.metadata)
    if save_filename:
        volume_save_path = Path(data_folder) / ("volume-" + save_filename)
        labels_save_path = Path(data_folder) / ("mask-" + save_filename)
        print(f"Saving volume to {volume_save_path} and labels to {labels_save_path}")
        synthetic_volume.write(volume_save_path)
        labels_ground_truth = grep_labels_from_directory(data_folder)
        labels = Nrrd(labels, labels_ground_truth.metadata)
        labels.write(labels_save_path)

    return synthetic_volume, labels
