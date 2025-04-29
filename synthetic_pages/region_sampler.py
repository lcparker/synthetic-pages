import numpy as np
import copy
from typing import Tuple, Union, List

from synthetic_pages.types.types import Indices3D, Point3D
from synthetic_pages.types.nrrd import Nrrd


def densely_subsample(
    input_volume: Nrrd,
    output_volume_size: Union[Point3D, Indices3D] = (256, 256, 256),
    samples: int = 8,
) -> List[Nrrd]:
    """
    Randomly sample multiple fixed-size subvolumes from a given Nrrd volume.
    Returns Nrrd objects with correct spatial metadata preserved.
    
    Args:
        input_volume: Nrrd object containing the volume to sample from
        output_volume_size: Desired size of each sampled volume (H, W, D)
        samples: Number of samples to generate
        prevent_overlap: If True, ensures samples don't overlap
    
    Returns:
        List[Nrrd]: List of Nrrd objects containing the sampled volumes with correct spatial metadata
    """
    input_shape = np.array(input_volume.volume.shape)
    output_size = np.array(output_volume_size)
    
    if any(output_size > input_shape):
        raise ValueError( f"Output volume size {output_size} cannot be larger than input volume size {input_shape}")
    
    valid_ranges = input_shape - output_size
    if any(valid_ranges < 0):
        raise ValueError( f"Input volume of size {input_shape} is too small to sample volumes of size {output_size}")
    
    result_nrrds = []
    space_directions = np.array(input_volume.metadata['space directions'])
    original_origin = np.array(input_volume.metadata['space origin'])
    
    while len(result_nrrds) < samples:
        # Generate random starting position
        start_pos = np.array([
            np.random.randint(0, valid_ranges[0] + 1),
            np.random.randint(0, valid_ranges[1] + 1),
            np.random.randint(0, valid_ranges[2] + 1)
        ])
        
        end_pos = start_pos + output_size
        
        # Extract the subvolume
        subvolume = input_volume.volume[
            start_pos[0]:end_pos[0],
            start_pos[1]:end_pos[1],
            start_pos[2]:end_pos[2]
        ]
        
        # Deep copy metadata with special handling for numpy arrays
        new_metadata = {}
        for key, value in input_volume.metadata.items():
            if isinstance(value, np.ndarray):
                new_metadata[key] = value.copy()
            else:
                new_metadata[key] = copy.deepcopy(value)
        
        # Update metadata
        new_metadata['sizes'] = list(subvolume.shape)
        
        # Calculate new origin using full space directions transformation
        offset = np.dot(space_directions.T, start_pos)
        new_origin = original_origin + offset
        new_metadata['space origin'] = new_origin
        
        # Create new Nrrd object
        result_nrrds.append(Nrrd(subvolume, new_metadata))
    
    return result_nrrds


