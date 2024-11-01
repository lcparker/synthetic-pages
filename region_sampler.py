import numpy as np
from typing import Tuple, Union
from main import Nrrd, Point3D, Indices3D
import numpy as np
from typing import Tuple, Union, List
from main import Nrrd, Point3D
import copy


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


import unittest
import numpy as np
from main import Nrrd
import tempfile
from pathlib import Path
from typing import List

class TestNrrdRandomSampling(unittest.TestCase):
    def setUp(self):
        """Create a sample Nrrd volume for testing"""
        # Create a simple gradient volume for easy validation
        volume = np.zeros((512, 512, 512))
        x, y, z = np.indices(volume.shape)
        volume = x + y + z  # Creates a gradient that increases in all directions
        
        # Create basic metadata
        self.metadata = {
            'type': 'float',
            'dimension': 3,
            'space': 'left-posterior-superior',
            'sizes': [512, 512, 512],
            'space directions': np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0]]),
            'kinds': ['domain', 'domain', 'domain'],
            'encoding': 'raw',
            'space origin': np.array([0.0, 0.0, 0.0])
        }
        
        self.test_nrrd = Nrrd(volume, self.metadata)
        
    def test_basic_sampling(self):
        """Test basic sampling functionality"""
        samples = densely_subsample(self.test_nrrd, (64, 64, 64), 5)
        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertEqual(sample.volume.shape, (64, 64, 64))
            self.assertTrue(isinstance(sample, Nrrd))

    def test_output_size_validation(self):
        """Test that the function raises an error for invalid output sizes"""
        with self.assertRaises(ValueError):
            densely_subsample(self.test_nrrd, (600, 64, 64), 1)

    def test_sample_boundaries(self):
        """Test that samples stay within the input volume bounds"""
        samples = densely_subsample(self.test_nrrd, (256, 256, 256), 10)
        input_max = np.max(self.test_nrrd.volume)
        for sample in samples:
            self.assertTrue(np.max(sample.volume) <= input_max)

    def test_spatial_metadata(self):
        """Test that spatial metadata is correctly preserved"""
        samples = densely_subsample(self.test_nrrd, (64, 64, 64), 3)
        for sample in samples:
            # Check metadata structure
            self.assertEqual(sample.metadata['dimension'], 3)
            self.assertEqual(sample.metadata['space'], 'left-posterior-superior')
            self.assertEqual(sample.metadata['sizes'], [64, 64, 64])
            
            # Check origin is within bounds of original volume
            origin = sample.metadata['space origin']
            max_origin = np.array(self.test_nrrd.metadata['space origin']) + \
                        np.array(self.test_nrrd.metadata['sizes']) - \
                        np.array([64, 64, 64])
            self.assertTrue(all(origin >= self.test_nrrd.metadata['space origin']))
            self.assertTrue(all(origin <= max_origin))

    def test_sample_content_validation(self):
        """Test that sampled content matches the expected region from the original volume"""
        sample_size = (32, 32, 32)
        samples = densely_subsample(self.test_nrrd, sample_size, 1)[0]
        
        # Get the origin in voxel coordinates
        origin = samples.metadata['space origin']
        origin_voxels = np.round(origin).astype(int)
        
        # Extract the same region directly from the original volume
        expected_content = self.test_nrrd.volume[
            origin_voxels[0]:origin_voxels[0]+sample_size[0],
            origin_voxels[1]:origin_voxels[1]+sample_size[1],
            origin_voxels[2]:origin_voxels[2]+sample_size[2]
        ]
        
        np.testing.assert_array_almost_equal(samples.volume, expected_content)

    def test_large_number_of_samples(self):
        """Test behavior with a large number of samples"""
        samples = densely_subsample(self.test_nrrd, (32, 32, 32), 100)
        self.assertEqual(len(samples), 100)

    def test_different_sample_sizes(self):
        """Test various sample size configurations"""
        test_sizes = [
            (32, 32, 32),
            (64, 32, 32),
            (32, 64, 32),
            (32, 32, 64),
            (16, 128, 64)
        ]
        for size in test_sizes:
            samples = densely_subsample(self.test_nrrd, size, 1)
            self.assertEqual(samples[0].volume.shape, size)

    def test_sample_persistence(self):
        """Test that samples can be saved and loaded correctly"""
        samples = densely_subsample(self.test_nrrd, (64, 64, 64), 1)[0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_file = Path(tmpdir) / "test_sample.nrrd"
            samples.write(temp_file)
            
            # Load the saved sample
            loaded_sample = Nrrd.from_file(temp_file)
            
            # Verify content and metadata
            np.testing.assert_array_equal(loaded_sample.volume, samples.volume)
            self.assertEqual(loaded_sample.metadata['space origin'].tolist(),
                           samples.metadata['space origin'].tolist())

    def test_anisotropic_voxel_spacing(self):
        """Test sampling with non-uniform voxel spacing"""
        # Create metadata with anisotropic spacing
        aniso_metadata = self.metadata.copy()
        aniso_metadata['space directions'] = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5]
        ])
        
        aniso_nrrd = Nrrd(self.test_nrrd.volume, aniso_metadata)
        samples = densely_subsample(aniso_nrrd, (64, 64, 64), 1)
        
        # Verify that spacing is preserved
        self.assertTrue(np.allclose(
            samples[0].metadata['space directions'],
            aniso_metadata['space directions']
        ))

if __name__ == "__main__":
    unittest.main()
