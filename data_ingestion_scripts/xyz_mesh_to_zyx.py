#!/usr/bin/env python3

"""
Transforms mesh coordinates from XYZ to ZYX frame.
Supports individual files or directories of .obj meshes.

XYZ (used in GP meshes) → ZYX (used in TIFFs / NRRDs).
"""

import numpy as np
from pathlib import Path
import argparse
from typing import List

from synthetic_pages.types.homogeneous_transform import HomogeneousTransform
from synthetic_pages.types.mesh import Mesh


def create_xyz_to_zyx_transform() -> HomogeneousTransform:
    matrix = np.array([
        [0, 0, 1, 0],  # X -> Z
        [0, 1, 0, 0],  # Y -> Y
        [1, 0, 0, 0],  # Z -> X
        [0, 0, 0, 1]
    ])
    return HomogeneousTransform(matrix)


def process_mesh(input_path: Path, output_path: Path) -> None:
    print(f"\nLoading mesh from {input_path}")
    mesh = Mesh.from_obj(str(input_path))

    print("Converting coordinates from XYZ to ZYX")
    transform = create_xyz_to_zyx_transform()
    transformed_mesh = transform.apply(mesh)

    print(f"Saving transformed mesh to {output_path}")
    transformed_mesh.to_obj(str(output_path))

    print("Mesh statistics:")
    print(f"  Vertices:  {len(mesh.points)}")
    print(f"  Triangles: {len(mesh.triangles)}")
    print(f"  Bounding box (before): min={mesh.points.min(axis=0)}, max={mesh.points.max(axis=0)}")
    print(f"  Bounding box (after):  min={transformed_mesh.points.min(axis=0)}, max={transformed_mesh.points.max(axis=0)}")


def find_obj_files(input_paths: List[Path]) -> List[Path]:
    obj_files: List[Path] = []
    for path in input_paths:
        if path.is_file() and path.suffix.lower() == '.obj':
            obj_files.append(path)
        elif path.is_dir():
            obj_files.extend(path.glob('*.obj'))
        else:
            print(f"Warning: {path} is not a valid .obj file or directory")
    return obj_files


def main() -> int:
    parser = argparse.ArgumentParser(description='Transform mesh coordinates from XYZ to ZYX frame')
    parser.add_argument('inputs', nargs='+', type=str, help='One or more .obj files or directories containing them')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory for transformed .obj files (filenames are preserved)')

    args = parser.parse_args()
    input_paths = [Path(p) for p in args.inputs]
    output_dir = Path(args.output_dir)

    obj_files = find_obj_files(input_paths)
    if not obj_files:
        print("No .obj files found.")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in obj_files:
        output_path = output_dir / input_path.name
        try:
            process_mesh(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

    print("\nAll transformations completed.")
    return 0


if __name__ == "__main__":
    exit(main())
