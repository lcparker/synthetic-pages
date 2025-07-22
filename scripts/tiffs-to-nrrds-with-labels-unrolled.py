from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional, Tuple, Iterator, List, Sequence, Dict
from dataclasses import dataclass
import gc
import logging
from concurrent.futures import ThreadPoolExecutor
import sys
import argparse

from synthetic_pages.mesh_operations import clip_mesh_to_bounding_box
from synthetic_pages.types.bounding_box_3d import BoundingBox
from synthetic_pages.types.mesh import Mesh
from synthetic_pages.types.nrrd import Nrrd

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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert TIFF stack to NRRD cubes')
parser.add_argument('input_dir', type=str, help='Input directory containing TIFF files')
parser.add_argument('output_dir', type=str, help='Output directory for NRRD files')
parser.add_argument('--min', type=int, nargs=3, required=True, 
                    help='Minimum coordinates as x y z', metavar=('X', 'Y', 'Z'))
parser.add_argument('--max', type=int, nargs=3, required=True,
                    help='Maximum coordinates as x y z', metavar=('X', 'Y', 'Z'))
parser.add_argument('--chunk-size', type=int, default=256, help='Size of cubic chunks')
parser.add_argument('--mesh-dir', type=str, default=None, help='Directory for mesh files representing sheets')
parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')

args = parser.parse_args()

if args.debug:
    logger.setLevel(logging.DEBUG)

# Setup paths and parameters
input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)
chunk_size = args.chunk_size
skip_existing = not args.force
mesh_dir = Path(args.mesh_dir) if args.mesh_dir else None

# Validate inputs
if not input_dir.is_dir():
    logger.error(f"Input directory {input_dir} does not exist")
    sys.exit(1)

# Create and validate bounding box
min_coords = args.min
max_coords = args.max

if len(min_coords) != 3 or len(max_coords) != 3:
    logger.error("Both min_coords and max_coords must have exactly 3 values")
    sys.exit(1)

# Ensure min is actually less than max for each dimension
for min_val, max_val, dim in zip(min_coords, max_coords, ['x', 'y', 'z']):
    if min_val >= max_val:
        logger.error(f"Min {dim} coordinate ({min_val}) must be less than max {dim} coordinate ({max_val})")
        sys.exit(1)

bbox = BoundingBox(
    x_start=min_coords[0],
    x_end=max_coords[0],
    y_start=min_coords[1],
    y_end=max_coords[1],
    z_start=min_coords[2],
    z_end=max_coords[2]
)

# Validate bounding box dimensions
bbox.validate(chunk_size)

# Load meshes if mesh directory is provided
meshes = []
if mesh_dir:
    for file in mesh_dir.glob("*.obj"):
        try:
            mesh = Mesh.from_obj(file)
            meshes.append(mesh)
            logger.info(f"Loading mesh from {file}")
        except Exception as e:
            logger.error(f"Failed to load mesh {file}: {str(e)}")

# Collect and validate all TIFF files
tiff_files = {}
for file_path in input_dir.glob("*.tif"):
    if file_path.is_file():
        # Extract the number from filename
        base = file_path.stem
        digits = ''.join(filter(str.isdigit, base))
        if not digits:
            logger.error(f"Could not extract number from filename: {file_path}")
            continue
        tiff_number = int(digits)
        tiff_files[tiff_number] = file_path

if not tiff_files:
    logger.error(f"No TIFF files found in {input_dir}")
    sys.exit(1)

logger.info(f"Found {len(tiff_files)} TIFF files with numbers from {min(tiff_files.keys())} to {max(tiff_files.keys())}")

# Process each z-chunk within the bounding box
for z_start in range(bbox.z_start, bbox.z_end, chunk_size):
    logger.info(f"Starting processing of z-chunk {z_start}")
    
    # Get files for this z-chunk
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
        continue
        
    if len(chunk_files) != chunk_size:
        logger.warning(f"Incomplete z-chunk {z_start}:{z_end}, found {len(chunk_files)} files, need {chunk_size}")
        continue
    
    # Generate all xy chunks for this z-chunk
    yx_chunks = []
    for y_start in range(bbox.y_start, bbox.y_end, chunk_size):
        for x_start in range(bbox.x_start, bbox.x_end, chunk_size):
            yx_chunks.append(ChunkIndices(
                x_start=x_start,
                x_end=x_start + chunk_size,
                y_start=y_start,
                y_end=y_start + chunk_size
            ))
    
    # Check if all output files for this z-chunk already exist
    if skip_existing:
        all_exist = True
        for chunk in yx_chunks:
            output_path = output_dir / f"{z_start:05d}_{chunk.y_start:05d}_{chunk.x_start:05d}_volume.nrrd"
            if not output_path.exists():
                all_exist = False
                break
        
        if all_exist:
            logger.info(f"Skipping z-chunk {z_start} - all output files exist")
            continue
    
    # Get image dimensions from first file
    try:
        with Image.open(chunk_files[0]) as img:
            height, width = img.height, img.width
    except Exception as e:
        logger.error(f"Failed to read image dimensions from {chunk_files[0]}: {str(e)}")
        continue
    
    logger.info(f"Processing volume of size (Z={chunk_size})x(Y={height})x(X={width})")
    logger.info(f"Bounding box: x={bbox.x_start}:{bbox.x_end}, y={bbox.y_start}:{bbox.y_end}")
    
    try:
        # Load TIFF stack
        if not chunk_files:
            logger.error("No TIFF files provided")
            continue
            
        # Load first image to get dimensions
        try:
            with Image.open(chunk_files[0]) as img:
                height, width = img.height, img.width
                image_array = np.array(img)
        except Exception as e:
            logger.error(f"Failed to load TIFF file {chunk_files[0]}: {str(e)}")
            continue
        
        logger.info(f"Loading TIFFS: Height/Y (axis 0) = {height}, Width/X (axis 1) = {width}")
        
        # Pre-allocate volume
        volume = np.zeros((len(chunk_files), bbox.y_end - bbox.y_start,bbox.x_end - bbox.x_start), dtype=np.uint16)
        volume[0] = image_array[bbox.y_start:bbox.y_end, bbox.x_start:bbox.x_end]
        
        # Load remaining images in parallel
        def load_image(idx, file_path, bbox):
            try:
                image = Image.open(file_path)
                if (image.height, image.width) != (height, width):
                    raise ValueError(f"Image dimensions {image_array.shape} don't match expected {(height, width)}")
                arr = np.array(image)
                image_array = arr[bbox.y_start:bbox.y_end, bbox.x_start:bbox.x_end]
                del image, arr
                        
                return idx, image_array
            except Exception as e:
                raise RuntimeError(f"Failed to load TIFF file {file_path}: {str(e)}")
        
        for i, f in enumerate(tqdm(chunk_files[1:], desc="Loading TIFF files")):
            idx, image_array = load_image(i+1, f, bbox)
            volume[idx] = image_array
        
        logger.info("TIFF stack loaded successfully")
        
        num_cubes = len(yx_chunks)
        logger.info(f"Processing {num_cubes} cubes for this z-chunk")
        
        # Process each xy chunk
        for chunk in tqdm(yx_chunks, desc="Processing cubes"):
            output_path = output_dir / f"{z_start:05d}_{chunk.y_start:05d}_{chunk.x_start:05d}_volume.nrrd"
            
            if skip_existing and output_path.exists():
                continue

            current_bounding_box = BoundingBox(
                    x_start=chunk.x_start,
                    x_end=chunk.x_end,
                    y_start=chunk.y_start,
                    y_end=chunk.y_end,
                    z_start=z_start,
                    z_end=z_end
                )

            any_intersections = False
            for mesh in meshes:
                if BoundingBox.intersection(mesh.bounding_box, current_bounding_box):
                    any_intersections = True
                    break
            if not any_intersections and meshes:
                logger.info(f"No meshes intersect with the current bounding box {current_bounding_box}, skipping...")
                continue

            mesh_sections = [
                clip_mesh_to_bounding_box(mesh, current_bounding_box) for mesh in meshes
            ]

            if len(mesh_sections) > 0 and not any(mesh_section for mesh_section in mesh_sections):
                logger.warning(f"No valid mesh sections for chunk {chunk} at z={z_start}, skipping...")
                continue
            
            # Extract cube from volume
            try:
                x_pos = chunk.x_start - bbox.x_start
                y_pos = chunk.y_start - bbox.y_start
                cube_data = volume[:, y_pos:y_pos + chunk_size, x_pos:x_pos + chunk_size]
            except Exception as e:
                logger.error(f"Failed to extract cube at coordinates {chunk}: {str(e)}")
                continue
            
            # Create NRRD from cube
            origin = (z_start, chunk.y_start, chunk.x_start)
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
            nrrd = Nrrd(cube_data, metadata)

            # Write NRRD file
            try:
                nrrd.write(output_path)
            except Exception as e:
                logger.error(f"Failed to write NRRD file {output_path}: {str(e)}")

            try:
                for i, mesh_section in enumerate(mesh_sections):
                    if mesh_section is not None:
                        mesh_output_path = output_path.parent / (output_path.stem + f"_mesh_{i}.obj")
                        mesh_section.to_obj(mesh_output_path)
                        logger.info(f"Wrote mesh section to {mesh_output_path}")
            except Exception as e:
                logger.error(f"Failed to write mesh sections for chunk {chunk}: {str(e)}")
                continue
        
        # Clean up memory
        del volume
        gc.collect()
        
    except Exception as e:
        logger.error(f"Failed to process z-chunk starting at {z_start}: {str(e)}")

logger.info("Processing complete")