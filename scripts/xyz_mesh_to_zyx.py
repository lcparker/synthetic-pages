#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse

from synthetic_pages.homogeneous_transform import HomogeneousTransform
from synthetic_pages.types.mesh import Mesh

def create_xyz_to_zyx_transform() -> HomogeneousTransform:
    """Creates transform matrix to convert from XYZ to ZYX"""
    matrix = np.array([
        [0, 0, 1, 0],  # X -> Z
        [0, 1, 0, 0],  # Y -> Y 
        [1, 0, 0, 0],  # Z -> X
        [0, 0, 0, 1]
    ])
    return HomogeneousTransform(matrix)

def process_mesh(input_path: str, output_path: str) -> None:
    """Load mesh, transform coordinates, and save"""
    print(f"Loading mesh from {input_path}")
    mesh = Mesh.from_obj(input_path)
    
    print("Converting coordinates from XYZ to ZYX")
    transform = create_xyz_to_zyx_transform()
    transformed_mesh = transform.apply(mesh)
    
    print(f"Saving transformed mesh to {output_path}")
    transformed_mesh.to_obj(output_path)
    
    # Print some stats
    print("\nMesh statistics:")
    print(f"Number of vertices: {len(mesh.points)}")
    print(f"Number of triangles: {len(mesh.triangles)}")
    print(f"\nBounding box before transform:")
    print(f"Min: {mesh.points.min(axis=0)}")
    print(f"Max: {mesh.points.max(axis=0)}")
    print(f"\nBounding box after transform:")
    print(f"Min: {transformed_mesh.points.min(axis=0)}")
    print(f"Max: {transformed_mesh.points.max(axis=0)}")

def main():
    parser = argparse.ArgumentParser(description='Transform mesh coordinates from XYZ to ZYX')
    parser.add_argument('input', type=str, help='Input OBJ file path')
    parser.add_argument('--output', '-o', type=str, help='Output OBJ file path (default: input_zyx.obj)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        # Create default output path by adding _zyx before the extension
        output_path = input_path.parent / f"{input_path.stem}_zyx{input_path.suffix}"
    
    try:
        process_mesh(str(input_path), str(output_path))
        print("\nTransformation completed successfully!")
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

