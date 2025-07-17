import unittest
import numpy as np
from pathlib import Path
import tempfile
import os

import trimesh

from synthetic_pages.types.bounding_box_3d import BoundingBox

class Mesh:
    def __init__(self, points, triangles):
        assert points.shape[-1] == 3
        assert len(points.shape) == 2
        assert triangles.shape[-1] == 3
        assert len(triangles.shape) == 2

        self.points = points
        self.triangles = triangles
        self.bounding_box = BoundingBox.from_min_max(self.points.min(axis=0), self.points.max(axis=0))

    def as_trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(vertices=self.points, faces=self.triangles, process=False)

    @staticmethod
    def from_trimesh(tm: trimesh.Trimesh) -> 'Mesh':
        if not isinstance(tm, trimesh.Trimesh):
            raise ValueError("Input must be a trimesh.Trimesh object")
        return Mesh(tm.vertices, tm.faces)

    def show(self,
            wireframe: bool = False,
            show: bool = True,
            *,
            fov_deg: float = 60.0,
            pad: float = 1.2,
            ):
        Mesh.display_meshes([self], wireframe=wireframe, show=show, fov_deg=fov_deg, pad=pad)

    @staticmethod
    def display_meshes(meshes: list['Mesh'],
            wireframe: bool = False,
            show: bool = True,
            *,
            fov_deg: float = 60.0,
            pad: float = 1.2,
            ):
        import trimesh

        if not meshes:
            raise ValueError("No meshes provided to show.")

        # Create a Trimesh Scene with all meshes
        scene = trimesh.Scene()
        for mesh in meshes:
            tm = trimesh.Trimesh(vertices=mesh.points, faces=mesh.triangles, process=False)
            scene.add_geometry(tm)

        if wireframe:
            edges = trimesh.load_path(np.concatenate([m.triangles for m in meshes]))
            scene.add_geometry(edges, geom_name="wire")

        scene.rezero()

        radius = tm.bounding_sphere.primitive.radius
        distance = pad * radius / np.sin(np.radians(fov_deg) / 2)

        scene.set_camera(distance=distance,
                        center=[0.0, 0.0, 0.0],
                        fov=[fov_deg, fov_deg])

        # widen clip planes to avoid accidental slicing
        scene.camera.z_far = 4 * distance
        scene.camera.z_near = max(distance / 1000, 1e-3)

        return scene.show(flags={'cull': False}) if show else scene

    @staticmethod
    def _stream_obj_elements(file_path: Path | str, chunk_size: int = 8192):
        """
        Stream elements from an OBJ file line by line using a generator.
        
        Args:
            file_path: Path to the OBJ file
            chunk_size: Size of buffer for reading file chunks
            
        Yields:
            tuple: (element_type, values) where element_type is 'v' or 'f' and values are the parsed numbers
        """
        remainder = ""
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    # Process any remaining data
                    if remainder:
                        line = remainder.strip()
                        if line and not line.startswith('#'):
                            values = line.split()
                            if values[0] in ('v', 'f'):
                                yield values[0], values[1:]
                    break
                
                # Decode chunk and combine with remainder
                text = remainder + chunk.decode('utf-8')
                lines = text.split('\n')
                
                # Save the last partial line for the next iteration
                remainder = lines[-1]
                
                # Process complete lines
                for line in lines[:-1]:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        values = line.split()
                        if values[0] in ('v', 'f'):
                            yield values[0], values[1:]

    @staticmethod
    def from_obj(file_path: Path | str) -> 'Mesh':
        """
        Read a mesh from an OBJ file using streaming.
        
        Args:
            file_path: Path to the OBJ file
            
        Returns:
            Mesh: A new Mesh instance
            
        Raises:
            ValueError: If the OBJ file is invalid or doesn't contain required data
        """
        # First pass: count vertices and faces
        num_vertices = 0
        num_faces = 0
        for elem_type, _ in Mesh._stream_obj_elements(file_path):
            if elem_type == 'v':
                num_vertices += 1
            elif elem_type == 'f':
                num_faces += 1
        
        # Preallocate arrays
        vertices = np.zeros((num_vertices, 3), dtype=np.float32)
        faces = np.zeros((num_faces, 3), dtype=np.int32)
        
        # Second pass: fill arrays
        v_idx = 0
        f_idx = 0
        for elem_type, values in Mesh._stream_obj_elements(file_path):
            if elem_type == 'v':
                try:
                    vertices[v_idx] = [float(x) for x in values[:3]]
                    v_idx += 1
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid vertex format: {values}")
            
            elif elem_type == 'f':
                try:
                    face = []
                    for v in values[:3]:  # Only take first three vertices for triangles
                        vertex_idx = int(v.split('/')[0]) - 1  # Convert to 0-based index
                        face.append(vertex_idx)
                    faces[f_idx] = face
                    f_idx += 1
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid face format: {values}")
        
        if len(vertices) == 0:
            raise ValueError("No vertices found in OBJ file")
        if len(faces) == 0:
            raise ValueError("No faces found in OBJ file")
        
        return Mesh(vertices, faces)

    def to_obj(self, file_path: Path | str, chunk_size: int = 8192) -> None:
        """
        Write the mesh to an OBJ file using streaming.
        
        Args:
            file_path: Path where to save the OBJ file
            chunk_size: Size of buffer for writing file chunks
        """
        def generate_obj_lines():
            # Header
            yield "# OBJ file created by Mesh class\n"
            
            # Vertices
            for vertex in self.points:
                yield f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n"
            
            # Faces (OBJ uses 1-based indexing)
            for face in self.triangles:
                yield f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"
        
        with open(file_path, 'w') as f:
            buffer = ""
            for line in generate_obj_lines():
                buffer += line
                if len(buffer) >= chunk_size:
                    f.write(buffer)
                    buffer = ""
            
            # Write any remaining data
            if buffer:
                f.write(buffer)



class TestMeshOBJ(unittest.TestCase):
    def setUp(self):
        # Create a simple cube mesh for testing
        self.points = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ], dtype=np.float32)
        
        self.triangles = np.array([
            [0, 1, 2],  # front
            [0, 2, 3],
            [1, 5, 6],  # right
            [1, 6, 2],
            [5, 4, 7],  # back
            [5, 7, 6],
            [4, 0, 3],  # left
            [4, 3, 7],
            [3, 2, 6],  # top
            [3, 6, 7],
            [0, 4, 5],  # bottom
            [0, 5, 1]
        ], dtype=np.int32)
        
        self.mesh = Mesh(self.points, self.triangles)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Clean up temporary files
        for file in Path(self.temp_dir).glob("*.obj"):
            file.unlink()
        os.rmdir(self.temp_dir)

    def test_save_and_load_simple(self):
        """Test basic save and load functionality"""
        file_path = Path(self.temp_dir) / "test_cube.obj"
        
        # Save mesh
        self.mesh.to_obj(file_path)
        
        # Load mesh
        loaded_mesh = Mesh.from_obj(file_path)
        
        # Compare vertices and faces
        np.testing.assert_array_almost_equal(self.mesh.points, loaded_mesh.points)
        np.testing.assert_array_equal(self.mesh.triangles, loaded_mesh.triangles)

    def test_streaming_large_mesh(self):
        """Test handling of larger meshes using streaming"""
        # Create a larger mesh
        n = 10000  # 10k vertices
        points = np.random.rand(n, 3).astype(np.float32)
        triangles = np.random.randint(0, n, (n*2, 3), dtype=np.int32)
        large_mesh = Mesh(points, triangles)
        
        file_path = Path(self.temp_dir) / "large_mesh.obj"
        
        # Save and load with streaming
        large_mesh.to_obj(file_path, chunk_size=1024)
        loaded_mesh = Mesh.from_obj(file_path)
        
        np.testing.assert_array_almost_equal(large_mesh.points, loaded_mesh.points)
        np.testing.assert_array_equal(large_mesh.triangles, loaded_mesh.triangles)

    def test_invalid_obj_format(self):
        """Test handling of invalid OBJ files"""
        file_path = Path(self.temp_dir) / "invalid.obj"
        
        # Create invalid OBJ file
        with open(file_path, 'w') as f:
            f.write("v 0 0\n")  # Invalid vertex (missing z)
            f.write("f 1 2\n")  # Invalid face (missing third vertex)
        
        with self.assertRaises(ValueError):
            Mesh.from_obj(file_path)

    def test_empty_obj(self):
        """Test handling of empty OBJ files"""
        file_path = Path(self.temp_dir) / "empty.obj"
        
        # Create empty OBJ file
        with open(file_path, 'w') as f:
            f.write("# Empty OBJ file\n")
        
        with self.assertRaises(ValueError):
            Mesh.from_obj(file_path)

    def test_obj_with_comments(self):
        """Test handling of OBJ files with comments and empty lines"""
        file_path = Path(self.temp_dir) / "commented.obj"
        
        # Create OBJ file with comments
        with open(file_path, 'w') as f:
            f.write("# This is a comment\n")
            f.write("\n")  # Empty line
            f.write("v 0 0 0\n")
            f.write("# Another comment\n")
            f.write("v 1 0 0\n")
            f.write("v 0 1 0\n")
            f.write("\n")
            f.write("f 1 2 3\n")
        
        mesh = Mesh.from_obj(file_path)
        self.assertEqual(len(mesh.points), 3)
        self.assertEqual(len(mesh.triangles), 1)

    def test_obj_with_texture_coords(self):
        """Test handling of OBJ files with texture coordinates"""
        file_path = Path(self.temp_dir) / "textured.obj"
        
        # Create OBJ file with texture coordinates
        with open(file_path, 'w') as f:
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 0 1 0\n")
            f.write("vt 0 0\n")  # Texture coordinates (should be ignored)
            f.write("vt 1 0\n")
            f.write("vt 0 1\n")
            f.write("f 1/1 2/2 3/3\n")  # Face with texture indices
        
        mesh = Mesh.from_obj(file_path)
        self.assertEqual(len(mesh.points), 3)
        self.assertEqual(len(mesh.triangles), 1)
        np.testing.assert_array_equal(mesh.triangles[0], [0, 1, 2])

    @classmethod
    def from_mask(
        cls, 
        mask: np.ndarray # (H, W, D)
        ):
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(mask, level=0)
        return Mesh(verts, faces)
