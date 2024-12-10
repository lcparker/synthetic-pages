from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from nrrd_file import Nrrd
from typing import Optional, Tuple, Iterator, List
from dataclasses import dataclass
import gc

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

    def validate(self, chunk_size: int = 256):
        """Ensure bounding box dimensions are valid for the given chunk size"""
        for start, end in [(self.x_start, self.x_end), 
                          (self.y_start, self.y_end), 
                          (self.z_start, self.z_end)]:
            if start % chunk_size != 0:
                raise ValueError(f"Start coordinate {start} must be a multiple of chunk_size {chunk_size}")
            if (end - start) % chunk_size != 0:
                raise ValueError(f"Bounding box dimension {end-start} must be a multiple of chunk_size {chunk_size}")

def get_image_dimensions(first_image_path: Path) -> Tuple[int, int]:
    with Image.open(first_image_path) as img:
        return img.height, img.width

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

def load_tiff_stack(tiff_files: List[Path]) -> np.ndarray:
    first_img = np.array(Image.open(tiff_files[0]))
    height, width = first_img.shape
    
    volume = np.zeros((len(tiff_files), height, width), dtype=first_img.dtype)
    volume[0] = first_img
    
    for i, file_path in enumerate(tqdm(tiff_files[1:], desc="Loading TIFF files")):
        image = Image.open(file_path)
        image_array = np.array(image)
        if image_array.shape != (height, width):
            raise ValueError(f"Inconsistent image dimensions in {file_path}")
        volume[i+1] = image_array
        del image
        del image_array
        gc.collect
    
    return volume

def extract_cube(
    volume: np.ndarray,
    chunk: ChunkIndices
) -> np.ndarray:
    return volume[:, chunk.y_start:chunk.y_end, chunk.x_start:chunk.x_end]

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

def process_tiff_stack(
    input_dir: Path,
    output_dir: Path,
    bbox: BoundingBox,
    chunk_size: int = 256
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Validate bounding box
    bbox.validate(chunk_size)
    
    # Get list of all TIFF files
    all_tiff_files = sorted([f for f in input_dir.glob("*.tif*") if f.is_file()])
    if not all_tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
        
    # Process each z-chunk within the bounding box
    for z_start in range(bbox.z_start, bbox.z_end, chunk_size):
        z_end = z_start + chunk_size
        z_slice = slice(z_start, z_end)
        
        
        # Check if we have enough files for this z-chunk
        if z_end > len(all_tiff_files):
            print(f"Warning: Not enough TIFF files for z-chunk {z_start}:{z_end}")
            continue
            
        tiff_files = all_tiff_files[z_slice]
        if len(tiff_files) != chunk_size:
            print(f"Warning: Incomplete z-chunk {z_start}:{z_end}, skipping")
            continue
            
        height, width = get_image_dimensions(tiff_files[0])
        print(f"Processing volume of size {width}x{height}x{chunk_size}")
        print(f"Bounding box: x={bbox.x_start}:{bbox.x_end}, y={bbox.y_start}:{bbox.y_end}")
        
        volume = load_tiff_stack(tiff_files)
        import matplotlib.pyplot as plt
        plt.imshow(volume[0], cmap='gray')
        plt.show()
        print("TIFF stack loaded successfully")
        
        xy_chunks = list(get_xy_chunks(bbox, chunk_size))
        num_cubes = len(xy_chunks)
        print(f"Processing {num_cubes} cubes for this z-chunk")
        
        for chunk in tqdm(xy_chunks, desc="Processing cubes"):
            cube_data = extract_cube(volume, chunk)
            
            origin = (chunk.x_start, chunk.y_start, z_start)
            nrrd = create_nrrd_from_cube(cube_data, origin)
            
            output_path = output_dir / f"{z_start:05d}_{chunk.y_start:05d}_{chunk.x_start:05d}_volume.nrrd"
            nrrd.write(output_path)
        
        # Free memory
        del volume

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert TIFF stack to NRRD cubes')
    parser.add_argument('input_dir', type=str, help='Input directory containing TIFF files')
    parser.add_argument('output_dir', type=str, help='Output directory for NRRD files')
    parser.add_argument('--x_start', type=int, required=True, help='Start X coordinate')
    parser.add_argument('--x_end', type=int, required=True, help='End X coordinate')
    parser.add_argument('--y_start', type=int, required=True, help='Start Y coordinate')
    parser.add_argument('--y_end', type=int, required=True, help='End Y coordinate')
    parser.add_argument('--z_start', type=int, required=True, help='Start Z coordinate')
    parser.add_argument('--z_end', type=int, required=True, help='End Z coordinate')
    parser.add_argument('--chunk_size', type=int, default=256, help='Size of cubic chunks')
    
    args = parser.parse_args()
    
    bbox = BoundingBox(
        x_start=args.x_start,
        x_end=args.x_end,
        y_start=args.y_start,
        y_end=args.y_end,
        z_start=args.z_start,
        z_end=args.z_end
    )
    
    process_tiff_stack(
        Path(args.input_dir),
        Path(args.output_dir),
        bbox,
        args.chunk_size
    )

if __name__ == "__main__":
    main()
