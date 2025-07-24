#!/usr/bin/env python3
"""
shaded_obj_viewer.py – fast, lit viewer for one or more large OBJ meshes.

Usage
-----
$ python shaded_obj_viewer.py mesh_1.obj mesh_2.obj …

With no arguments it prints a short usage message.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

import open3d as o3d
from open3d.visualization.rendering import MaterialRecord


def random_rgba() -> List[float]:
    """Return a random opaque RGBA quadruplet in [0, 1] for per-mesh tinting."""
    return [random.random(), random.random(), random.random(), 1.0]


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    """Load an OBJ and ensure per-vertex normals are available for lighting."""
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive viewer with Phong lighting for very large OBJ meshes "
            "(≈100 MB each). Requires Open3D ≥ 0.17."
        )
    )
    parser.add_argument(
        "meshes",
        metavar="OBJ",
        type=Path,
        nargs="+",
        help="path(s) to .obj mesh file(s)",
    )
    args = parser.parse_args()

    geometries: List[dict] = []
    for index, path in enumerate(args.meshes):
        mesh = load_mesh(path)

        # Create a lightweight material that keeps lighting enabled.
        material = MaterialRecord()
        material.shader = "defaultLit"           # Phong/Blinn shading
        material.base_color = random_rgba()      # per-mesh tint
        material.base_roughness = 0.9            # slightly matte

        geometries.append(
            {
                "name": f"{path.name}-{index}",
                "geometry": mesh,
                "material": material,
            }
        )
        print(f"Loaded mesh {path.name}")

    o3d.visualization.draw(
        geometries,
        title="Shaded OBJ preview",
        show_skybox=False,            # neutral grey background
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python shaded_obj_viewer.py <mesh1.obj> [<mesh2.obj> …]")
        sys.exit(1)

    main()

