from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from synthetic_pages.types.nrrd import Nrrd
from typing import Optional, Tuple, Iterator, List, Sequence, Dict
from dataclasses import dataclass
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class ChunkIndices:
    x_start: int
    x_end: int
    y_start: int
    y_end: int

@dataclass
class BoundingBox:
    x_start: int
    x_end: int
    y_start: int
    y_end: int
    z_start: int
    z_end: int

    @classmethod
    def from_min_max(cls, min_coords: Sequence[int], max_coords: Sequence[int]):
        if len(min_coords) != 3 or len(max_coords) != 3:
            raise ValueError("Both min_coords and max_coords must have exactly 3 values")
        
        # Ensure min is actually less than max for each dimension
        for min_val, max_val, dim in zip(min_coords, max_coords, ['x', 'y', 'z']):
            if min_val >= max_val:
                raise ValueError(f"Min {dim} coordinate ({min_val}) must be less than max {dim} coordinate ({max_val})")
        
        return cls(
            x_start=min_coords[0],
            x_end=max_coords[0],
            y_start=min_coords[1],
            y_end=max_coords[1],
            z_start=min_coords[2],
            z_end=max_coords[2]
        )

    def validate(self, chunk_size: int = 256):
        """Ensure bounding box dimensions are valid for the given chunk size"""
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
            
        for dim, (start, end) in enumerate([
            (self.x_start, self.x_end),
            (self.y_start, self.y_end),
            (self.z_start, self.z_end)
        ], 1):
            if start < 0:
                raise ValueError(f"Dimension {dim} start coordinate ({start}) cannot be negative")
                
            if start % chunk_size != 0:
                raise ValueError(f"Dimension {dim} start coordinate ({start}) must be a multiple of chunk_size {chunk_size}")
                
            if (end - start) % chunk_size != 0:
                raise ValueError(f"Dimension {dim} size ({end-start}) must be a multiple of chunk_size {chunk_size}")

def get_image_dimensions(first_image_path: Path) -> Tuple[int, int]:
    try:
        with Image.open(first_image_path) as img:
            return img.height, img.width
    except Exception as e:
        raise RuntimeError(f"Failed to read image dimensions from {first_image_path}: {str(e)}")

def get_xy_chunks(bbox: BoundingBox, chunk_size: int = 256) -> Iterator[ChunkIndices]:
    """Generate x,y coordinate chunks within the bounding box"""
    for y_start in range(bbox.y_start, bbox.y_end, chunk_size):
        for x_start in range(bbox.x_start, bbox.x_end, chunk_size):
            yield ChunkIndices(
                x_start=x_start,
                x_end=x_start + chunk_size,
                y_start=y_start,
                y_end=y_start + chunk_size
            )

def load_single_tiff(file_path: Path, expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load a single TIFF file with error handling and validation"""
    try:
        with Image.open(file_path) as image:
            image_array = np.array(image).T
            
            if expected_shape and image_array.shape != expected_shape:
                raise ValueError(f"Image dimensions {image_array.shape} don't match expected {expected_shape}")
                
            return image_array
    except Exception as e:
        raise RuntimeError(f"Failed to load TIFF file {file_path}: {str(e)}")

def load_tiff_stack(tiff_files: List[Path]) -> np.ndarray:
    if not tiff_files:
        raise ValueError("No TIFF files provided")
        
    # Load first image to get dimensions
    first_img = load_single_tiff(tiff_files[0])
    height, width = first_img.shape
    logger.info(f"Loading TIFFS: Height (axis 0) = {height}, Width (axis 1) = {width}")
    
    # Pre-allocate volume
    volume = np.zeros((len(tiff_files), height, width), dtype=first_img.dtype)
    volume[0] = first_img
    
    # Load remaining images in parallel
    def load_image(args):
        idx, file_path = args
        return idx, load_single_tiff(file_path, (height, width))
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_image, (i+1, f)) for i, f in enumerate(tiff_files[1:])]
        
        for future in tqdm(futures, desc="Loading TIFF files"):
            idx, image_array = future.result()
            volume[idx] = image_array
            
    return volume

def extract_cube(
    volume: np.ndarray,
    chunk: ChunkIndices
) -> np.ndarray:
    try:
        return volume[:, chunk.y_start:chunk.y_end, chunk.x_start:chunk.x_end]
    except Exception as e:
        raise RuntimeError(f"Failed to extract cube at coordinates {chunk}: {str(e)}")

def create_nrrd_from_cube(cube_data: np.ndarray, origin: tuple[int, int, int]) -> Nrrd:
    metadata = {
        'type': 'int16',
        'dimension': 3,
        'space': 'left-posterior-superior',
        'sizes': cube_data.shape,
        'space origin': np.array(origin).astype(float),
        'kinds': ['domain', 'domain', 'domain'],
        'endian': 'little',
        'encoding': 'gzip',
        'space directions': np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
    }
    return Nrrd(cube_data, metadata)

def get_tiff_number(file_path: Path) -> int:
    """Extract the number from a TIFF filename"""
    base = file_path.stem
    digits = ''.join(filter(str.isdigit, base))
    if not digits:
        raise ValueError(f"Could not extract number from filename: {file_path}")
    return int(digits)

def collect_tiff_files(input_dir: Path) -> Dict[int, Path]:
    """Collect and validate all TIFF files from the input directory"""
    tiff_files = {
        get_tiff_number(f): f 
        for f in input_dir.glob("*.tif*") 
        if f.is_file()
    }
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
    
    logger.info(f"Found {len(tiff_files)} TIFF files with numbers from {min(tiff_files.keys())} to {max(tiff_files.keys())}")
    return tiff_files

def get_chunk_files(
    tiff_files: Dict[int, Path],
    z_start: int,
    chunk_size: int
) -> Optional[List[Path]]:
    """Get the list of files needed for a specific z-chunk"""
    z_end = z_start + chunk_size
    needed_numbers = range(z_start, z_end)
    chunk_files = []
    missing_files = []
    
    for num in needed_numbers:
        if num in tiff_files:
            chunk_files.append(tiff_files[num])
        else:
            missing_files.append(num)
    
    if missing_files:
        logger.warning(f"Missing TIFF files for numbers: {missing_files}")
        return None
        
    if len(chunk_files) != chunk_size:
        logger.warning(f"Incomplete z-chunk {z_start}:{z_end}, found {len(chunk_files)} files, need {chunk_size}")
        return None
        
    return chunk_files

def check_existing_outputs(
    output_dir: Path,
    z_start: int,
    xy_chunks: List[ChunkIndices]
) -> bool:
    """Check if all output files for a z-chunk already exist"""
    for chunk in xy_chunks:
        output_path = output_dir / f"{z_start:05d}_{chunk.y_start:05d}_{chunk.x_start:05d}_volume.nrrd"
        if not output_path.exists():
            return False
    return True

def process_xy_chunk(
    volume: np.ndarray,
    chunk: ChunkIndices,
    z_start: int,
    output_dir: Path,
    skip_existing: bool = True
) -> None:
    """Process a single xy chunk and save it as an NRRD file"""
    output_path = output_dir / f"{z_start:05d}_{chunk.y_start:05d}_{chunk.x_start:05d}_volume.nrrd"
    
    if skip_existing and output_path.exists():
        return
        
    cube_data = extract_cube(volume, chunk)
    origin = (z_start, chunk.y_start, chunk.x_start)
    nrrd = create_nrrd_from_cube(cube_data, origin)
    
    try:
        nrrd.write(output_path)
    except Exception as e:
        logger.error(f"Failed to write NRRD file {output_path}: {str(e)}")

def process_z_chunk(
    tiff_files: List[Path],
    bbox: BoundingBox,
    z_start: int,
    chunk_size: int,
    output_dir: Path,
    skip_existing: bool = True
) -> None:
    """Process a single z-chunk of TIFF files"""
    width, height = get_image_dimensions(tiff_files[0])
    logger.info(f"Processing volume of size {width}x{height}x{chunk_size}")
    logger.info(f"Bounding box: x={bbox.x_start}:{bbox.x_end}, y={bbox.y_start}:{bbox.y_end}")
    
    try:
        volume = load_tiff_stack(tiff_files)
        logger.info("TIFF stack loaded successfully")
        
        xy_chunks = list(get_xy_chunks(bbox, chunk_size))
        num_cubes = len(xy_chunks)
        logger.info(f"Processing {num_cubes} cubes for this z-chunk")
        
        for chunk in tqdm(xy_chunks, desc="Processing cubes"):
            process_xy_chunk(volume, chunk, z_start, output_dir, skip_existing)
            
        del volume
        gc.collect()
        
    except Exception as e:
        logger.error(f"Failed to process z-chunk starting at {z_start}: {str(e)}")

def process_tiff_stack(
    input_dir: Path,
    output_dir: Path,
    bbox: BoundingBox,
    chunk_size: int = 256,
    skip_existing: bool = True
) -> None:
    """Main processing function to convert TIFF stack to NRRD cubes"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Validate inputs
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    bbox.validate(chunk_size)
    
    # Collect and validate all TIFF files
    tiff_files = collect_tiff_files(input_dir)
    
    # Process each z-chunk within the bounding box
    for z_start in range(bbox.z_start, bbox.z_end, chunk_size):
        logger.info(f"Starting processing of z-chunk {z_start}")
        
        # Get files for this chunk
        chunk_files = get_chunk_files(tiff_files, z_start, chunk_size)
        if not chunk_files:
            continue
            
        # Check if outputs exist
        xy_chunks = list(get_xy_chunks(bbox, chunk_size))
        if skip_existing and check_existing_outputs(output_dir, z_start, xy_chunks):
            logger.info(f"Skipping z-chunk {z_start} - all output files exist")
            continue
            
        # Process this z-chunk
        process_z_chunk(chunk_files, bbox, z_start, chunk_size, output_dir, skip_existing)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert TIFF stack to NRRD cubes')
    parser.add_argument('input_dir', type=str, help='Input directory containing TIFF files')
    parser.add_argument('output_dir', type=str, help='Output directory for NRRD files')
    parser.add_argument('--min', type=int, nargs=3, required=True, 
                        help='Minimum coordinates as x y z', metavar=('X', 'Y', 'Z'))
    parser.add_argument('--max', type=int, nargs=3, required=True,
                        help='Maximum coordinates as x y z', metavar=('X', 'Y', 'Z'))
    parser.add_argument('--chunk-size', type=int, default=256, help='Size of cubic chunks')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    try:
        bbox = BoundingBox.from_min_max(args.min, args.max)
        process_tiff_stack(
            Path(args.input_dir),
            Path(args.output_dir),
            bbox,
            args.chunk_size,
            skip_existing=not args.force
        )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
