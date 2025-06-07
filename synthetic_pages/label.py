from pathlib import Path
from dataclasses import dataclass
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QHBoxLayout, QWidget, QDialog, QLabel, QFileDialog, QListWidget, QListWidgetItem, QShortcut, QSizePolicy)
from PyQt5.QtGui import QKeySequence
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from PyQt5.QtCore import Qt
from synthetic_pages.types.nrrd import Nrrd
from vtkmodules.util.numpy_support import numpy_to_vtk
from synthetic_pages.match_stitches import match_stitches

@dataclass
class Segmentation:
    name: str 
    volume: Nrrd | None
    masks: list[Nrrd]
    volume_actor: vtk.vtkVolume | None
    visible: bool

class Model:
    segmentations: list[Segmentation] = []

class SegmentationManager:
    model: Model

    def __init__(self, model = Model()):
        self.model = model

    def load_segmentation(self, name, volume_path, mask_path) -> None:
        try:
            mask = Nrrd.from_file(mask_path)
            new_segmentation = Segmentation(
                    name, 
                    Nrrd.from_file(volume_path) if volume_path else None,
                    [mask],
                    None,
                    True)
            
            self.model.segmentations.append(new_segmentation)
 
            
        except Exception as e:
            print(f"Error loading files: {str(e)}")

    def get_segmentation_by_index(self, idx: int):
        idx = int(idx)
        if self.number_of_segmentations() < idx:
            raise ValueError(f"No segmentation exists with index {idx}")
        return self.segmentations[idx]

    def get_segmentation_by_actor(self, actor: vtk.vtkProp3D | vtk.vtkActor | None) -> Segmentation | None:
        if not actor:
            return None
        for seg in self.segmentations:
            if seg.visible and seg.volume_actor == actor:
                return seg

        return None

    def load_segmentations(self, mask_filenames: list[str]) -> None:
        if mask_filenames:
            for p in mask_filenames:
                try:
                    # Parse coordinates from filename
                    filename = p.split('/')[-1]
                    if not filename.endswith('_mask.nrrd'):
                        print(f"Skipping {filename} - not a mask file")
                        continue
                    
                    # Extract coordinates and format with leading zeros
                    coords = filename.replace('_mask.nrrd', '').split('_')
                    if len(coords) != 3:
                        print(f"Skipping {filename} - invalid format")
                        continue
                    
                    z, y, x = [f"{int(coord):05d}" for coord in coords]
                    # Keep Z,Y,X order in the display name
                    name = f"Z{z}_Y{y}_X{x}"  # Changed order here
                    
                    volume_path = p.replace('_mask.nrrd', '_volume.nrrd')
                    if not Path(volume_path).exists():
                        volume_path = None
                    
                    self.load_segmentation(name, volume_path, p)
                    
                    print(f"Successfully loaded mask for {name}")
                    
                except Exception as e:
                    print(f"Error loading {p}: {str(e)}")

    def has_modification(self) -> bool:
        return any(len(s.masks) > 0 for s in self.segmentations)

    def save_updated_segmentations(self, save_dir: Path) -> None:
        if save_dir:
            try:
                save_dir = Path(save_dir)
                saved_count = 0
                
                for seg in self.segmentations:
                    if len(seg.masks) > 1:
                        mask = seg.masks[-1]
                        z, y, x = mask.metadata['space origin']
                        
                        # Save most recent modification
                        new_filename = f"{int(z):05d}_{int(y):05d}_{int(x):05d}_mask.nrrd"
                        save_path = save_dir / new_filename
                        mask.write(save_path)
                        saved_count += 1
                
                print(f"Successfully saved {saved_count} modified segmentations to {save_dir}")
                    
            except Exception as e:
                print(f"Error saving segmentations: {str(e)}")


    @property
    def segmentations(self) -> list[Segmentation]:
        return self.model.segmentations

    def number_of_segmentations(self) -> int:
        return len(self.model.segmentations)

class SegmentationVisualizer:
    def __init__(self, renderer: vtk.vtkRenderer, segmentation_manager: SegmentationManager):
        self.renderer = renderer
        self.segmentation_manager = segmentation_manager
        self.color_map = self.initialize_color_map()

    def visualize_segmentation(self, segmentation, state=None):
        pass

    def update_visualization(self, reset_camera=False):
        self.renderer.RemoveAllViewProps()
        
        for segmentation in self.segmentation_manager.segmentations:
            if segmentation.visible:
                segmentation.volume_actor = self.wrap_segmentation_in_actor(segmentation)
                self.renderer.AddVolume(segmentation.volume_actor)
            
        if reset_camera:
            self.renderer.ResetCamera()
            
        self.renderer.GetRenderWindow().Render()

    def wrap_segmentation_in_actor(self, segmentation: Segmentation) -> vtk.vtkVolume:
        # Convert from NRRD standard ordering (ZYX) to VTK's expected ordering (XYZ)
        mask_data = segmentation.masks[-1].volume.T
        metadata = segmentation.masks[-1].metadata
        
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(mask_data.shape)
        vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        vtk_array = numpy_to_vtk(mask_data.flatten(), deep=True)
        vtk_data.GetPointData().GetScalars().DeepCopy(vtk_array)
        color_tf = self.color_map
        
        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(0, 0.0)  # Air is transparent
        opacity_tf.AddPoint(1, 0.8)
        
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_tf)
        volume_property.SetScalarOpacity(opacity_tf)
        volume_property.ShadeOn()
        
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_data)
        
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(volume_property)
        
        matrix = vtk.vtkMatrix4x4()
        space_directions = metadata['space directions']
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, space_directions[i][j])
        
        space_origin = metadata['space origin']
        for i in range(3):
            matrix.SetElement(i, 3, space_origin[i])
        
        volume.SetUserMatrix(matrix)

        return volume


    def initialize_color_map(self) -> vtk.vtkColorTransferFunction:
        """Create a consistent color mapping using the curated color set"""
        color_map = vtk.vtkColorTransferFunction()
        
        # Make 0 (air) black/transparent
        color_map.AddRGBPoint(0, 0, 0, 0)
        
        # Curated color palette - values in RGB (0-1 range)
        base_colors = [
            (0.8941, 0.1020, 0.1098),  # Coral Red
            (0.2157, 0.4941, 0.7216),  # Steel Blue
            (0.3019, 0.6863, 0.2902),  # Forest Green
            (0.5961, 0.3059, 0.6392),  # Amethyst Purple
            (1.0000, 0.4980, 0.0000),  # Orange
            (0.2275, 0.7294, 0.6235),  # Turquoise
            (0.9059, 0.5412, 0.7647),  # Rose Pink
            (0.4000, 0.6510, 0.1176),  # Apple Green
            (0.9412, 0.8941, 0.2588),  # Sunshine Yellow
            (0.3373, 0.7059, 0.9137),  # Sky Blue
            (0.8392, 0.3765, 0.3020),  # Salmon
            (0.4941, 0.3137, 0.6510),  # Royal Purple
            (0.6235, 0.6000, 0.4392),  # Warm Gray
            (0.3176, 0.4784, 0.2078),  # Olive Green
            (0.8431, 0.6275, 0.3098)   # Golden Brown
        ]
        
        num_base_colors = len(base_colors)
        
        # Add colors to the transfer function
        for i in range(1, 32):  # 1-31 for non-air segments
            if i < num_base_colors:
                # Use base colors for first set of segments
                r, g, b = base_colors[i]
            else:
                # For additional segments, create variations of base colors
                base_idx = i % num_base_colors
                variation = (i // num_base_colors) + 1
                
                # Create a slightly different shade of the base color
                r, g, b = base_colors[base_idx]
                # Adjust brightness and saturation based on variation
                factor = 1.0 / (1.0 + variation * 0.3)
                r = max(0.1, min(1.0, r * factor))
                g = max(0.1, min(1.0, g * factor))
                b = max(0.1, min(1.0, b * factor))
            
            color_map.AddRGBPoint(i, r, g, b)

        return color_map


from abc import ABC, abstractmethod
class IHandlePick(ABC):
    @abstractmethod
    def on_click(self, obj, event):
        raise NotImplementedError()

    @abstractmethod
    def activate(self):
        raise NotImplementedError()

    @abstractmethod
    def deactivate(self):
        raise NotImplementedError()

class PickingHandler(IHandlePick):
    picker: vtk.vtkCellPicker
    segmentation_manager: SegmentationManager
    visualizer: SegmentationVisualizer
    giver_selected: Segmentation|None
    active: bool

    @property
    def renderer(self) -> vtk.vtkRenderer:
        return self.visualizer.renderer

    @property
    def render_window_interactor(self) -> vtk.vtkRenderWindowInteractor:
        return self.renderer.GetRenderWindow().GetInteractor()

    def __init__(self, segmentation_manager: SegmentationManager, visualizer: SegmentationVisualizer):
        # Set up picker
        self.giver_selected = None
        self.picker = vtk.vtkCellPicker()
        self.segmentation_manager = segmentation_manager
        self.visualizer = visualizer
        self.active = False

    def activate(self):
        self.visualizer.renderer.GetRenderWindow().GetInteractor().SetPicker(self.picker)
        self.active=True

    def deactivate(self):
        self.visualizer.renderer.GetRenderWindow().GetInteractor().SetPicker(None)
        self.active=False


    def on_click(self, obj, event):
        if not self.active:
            return

        click_pos = self.render_window_interactor.GetEventPosition()
        pick = self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        if not pick:
            return

        picked_volume = self.picker.GetProp3D()
        picked_segmentation = self.segmentation_manager.get_segmentation_by_actor(picked_volume)

        if not picked_segmentation:
            return
        if not self.giver_selected:
            self.giver_selected = picked_segmentation
            return
        elif picked_segmentation == self.giver_selected:
            print("Cannot select same volume as giver and receiver")
            self.giver_selected=None
            return
                
        try:
            if not self.giver_selected:
                raise Exception("No giver segmentation was selected while trying to select receiver")
            updated_segmentation = match_stitches(self.giver_selected.masks[-1], 
                                       picked_segmentation.masks[-1])
            picked_segmentation.masks.append(updated_segmentation)
            print(f"Successfully matched labels {self.giver_selected.name} -> {picked_segmentation.name}")
        except Exception as e:
            print(f"Error matching labels: {str(e)}")
        finally:
            self.giver_selected = None
        
        self.visualizer.update_visualization()


import numpy as np
class LayerPickingHandler(IHandlePick):
    def __init__(self, segmentation_manager: SegmentationManager, visualizer: SegmentationVisualizer):
        self.picker = vtk.vtkCellPicker()
        self.segmentation_manager = segmentation_manager
        self.visualizer = visualizer
        self.active = False
        self.original_nrrds = {}
        
    def traverse_ray_dda(self, start_indices, ray_dir, volume):
        position = np.array(start_indices, dtype=np.float64)
        step = ray_dir / np.linalg.norm(ray_dir)  # Normalize step size
        step /=2
        
        z_dim, y_dim, x_dim = volume.shape
        max_steps = int(np.ceil(np.sqrt(x_dim*x_dim + y_dim*y_dim + z_dim*z_dim)))
        for _ in range(max_steps):
            x,y,z = np.round(position).astype(int)
            
            if (0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim):
                value = volume[x, y, z]
                # world_point = self.volume_to_world_coords((x, y, z), volume_matrix)
                print(f"value at ({x}, {y}, {z}) is {value}")
                if value > 0:
                    return (x, y, z), value
                    
            position += step
            
            # Check if we're completely outside the volume
            if np.any(position < 0) or np.any(position >= np.array([x_dim, y_dim, z_dim])):
                break
                
        return None, None

    @property
    def renderer(self) -> vtk.vtkRenderer:
        return self.visualizer.renderer

    @property
    def render_window_interactor(self) -> vtk.vtkRenderWindowInteractor:
        return self.renderer.GetRenderWindow().GetInteractor()

    def activate(self):
        self.render_window_interactor.SetPicker(self.picker)
        self.active = True
        print("Layer picking active - click to isolate a layer")

    def deactivate(self):
        for seg_id, original_nrrd in self.original_nrrds.items():
            for seg in self.segmentation_manager.segmentations:
                if seg.name == seg_id:
                    seg.masks[-1] = original_nrrd
                    break
        self.original_nrrds.clear()
        self.render_window_interactor.SetPicker(None)
        self.active = False
        self.visualizer.update_visualization()
        print("Layer picking deactivated - restored original view")

    def on_click(self, obj, event):
        if not self.active:
            return

        click_pos = self.render_window_interactor.GetEventPosition()
        pick = self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)

        if not pick:
            return

        picked_volume = self.picker.GetProp3D()
        picked_segmentation = self.segmentation_manager.get_segmentation_by_actor(picked_volume)

        if not picked_segmentation:
            return

        # Get camera position and picked position in world coordinates
        camera = self.renderer.GetActiveCamera()
        camera_pos = camera.GetPosition()
        picked_position = self.picker.GetPickPosition()

        # Calculate a point far along the ray for visualization
        direction = np.array(picked_position) - np.array(camera_pos)
        direction = direction / np.linalg.norm(direction)

        current_nrrd = picked_segmentation.masks[-1]
        mask_volume = current_nrrd.volume
        volume_matrix = picked_volume.GetUserMatrix()

        try:
            # Get indices of picked point first
            picked_indices = self.world_to_volume_coords(picked_position, volume_matrix)
            x, y, z = np.round(picked_indices).astype(int)
            
            dir_end = np.array(picked_position) + direction
            dir_end_volume = self.world_to_volume_coords(dir_end, volume_matrix)
            direction_volume = np.array(dir_end_volume) - np.array(picked_indices)
            direction_volume = direction_volume / np.linalg.norm(direction_volume)

            if 0 <= x < mask_volume.shape[2] and 0 <= y < mask_volume.shape[1] and 0 <= z < mask_volume.shape[0]:
                value = mask_volume[x, y, z]
                print(f"Value at picked point: {value}")
                
                if value == 0:
                    print("Initial pick point is in air, traversing ray...")
                    indices, value = self.traverse_ray_dda(picked_indices, direction_volume, mask_volume)
                    
                    if indices is not None:
                        print(f"Found non-zero value {value} at position {indices}")
                        
                        if picked_segmentation.name not in self.original_nrrds:
                            self.original_nrrds[picked_segmentation.name] = current_nrrd
                        
                        # Create a new volume showing only the selected layer
                        new_volume = np.zeros_like(mask_volume)
                        new_volume[mask_volume == value] = value
                        picked_segmentation.masks[-1] = Nrrd(new_volume, current_nrrd.metadata.copy())
                        self.visualizer.update_visualization()
                    else:
                        print("No non-zero value found along ray")
                else:
                    self.visualizer.update_visualization()
                    print(f"Found non-zero value {value} at initial pick point")

        except Exception as e:
            print(f"Error checking picked point: {e}")

    def world_to_volume_coords(self, point, volume_matrix):
        inverse_matrix = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(volume_matrix, inverse_matrix)
        p = list(point) + [1.0]
        res = [0, 0, 0, 0]
        inverse_matrix.MultiplyPoint(p, res)
        return res[:3]

    
    def volume_to_world_coords(self, point, volume_matrix):
        p = list(point) + [1.0]
        res = [0, 0, 0, 0]
        volume_matrix.MultiplyPoint(p, res)
        return res[:3]


class PickingInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        self.picking_enabled = False
        
    def OnLeftButtonDown(self, obj, event):
        if not self.picking_enabled:
            super().OnLeftButtonDown(obj, event)
            
    def OnLeftButtonUp(self, obj, event):
        if not self.picking_enabled:
            super().OnLeftButtonUp(obj, event)
            
    def OnMouseMove(self, obj, event):
        if not self.picking_enabled:
            super().OnMouseMove(obj, event)
            
    def OnMouseWheelForward(self, obj, event):
        if not self.picking_enabled:
            super().OnMouseWheelForward(obj, event)
            
    def OnMouseWheelBackward(self, obj, event):
        if not self.picking_enabled:
            super().OnMouseWheelBackward(obj, event)
            
    def EnablePicking(self):
        self.picking_enabled = True
        
    def DisablePicking(self):
        self.picking_enabled = False
        
class MainWindow(QMainWindow):
    segmentation_manager: SegmentationManager

    @property
    def next_seg_id(self) -> int:
        return self.segmentation_manager.number_of_segmentations() + 1

    def __init__(self, parent=None, segmentation_manager = SegmentationManager()):
        super(MainWindow, self).__init__(parent)
        self.segmentation_manager = segmentation_manager
        
        # Create main layout
        self.central_widget = QWidget()
        main_layout = QHBoxLayout(self.central_widget)
        left_panel = self.__create_left_panel()
        main_layout.addWidget(left_panel)
        
        # Create the VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor()
        main_layout.addWidget(self.vtk_widget)

        left_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.vtk_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Set up VTK renderer
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()


        style = PickingInteractorStyle()
        style.SetDefaultRenderer(self.renderer)
        self.iren.SetInteractorStyle(style)


        self.visualizer = SegmentationVisualizer(self.renderer, self.segmentation_manager)
        
        self.picking_handler  = PickingHandler(self.segmentation_manager, self.visualizer)
        self.layer_picking_handler = LayerPickingHandler(self.segmentation_manager, self.visualizer)
        self.iren.AddObserver("LeftButtonPressEvent", self.handle_click)
        
        self.setup_keypress_callbacks()
        
        self.setCentralWidget(self.central_widget)
        self.show()
        self.iren.Initialize()

    def setup_keypress_callbacks(self):
        self.iren.AddObserver('KeyPressEvent', self.keypress_callback)

        self.quit_shortcut = QShortcut(QKeySequence('q'), self)
        self.quit_shortcut.activated.connect(self.close_application)

        self.load_shortcut = QShortcut(QKeySequence('l'), self)
        self.load_shortcut.activated.connect(self.load_multiple_masks)

        self.match_shortcut = QShortcut(QKeySequence('m'), self)
        self.match_shortcut.activated.connect(self.toggle_label_matching)

        self.layer_pick_shortcut = QShortcut(QKeySequence('p'), self)
        self.layer_pick_shortcut.activated.connect(self.toggle_layer_picking)


    def __create_left_panel(self) -> QWidget:
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.load_multiple_btn = QPushButton("Load Multiple Masks")
        self.load_multiple_btn.clicked.connect(self.load_multiple_masks)
        left_layout.addWidget(self.load_multiple_btn)
        
        # Create list of loaded segmentations
        self.segmentation_visibility_list = QListWidget()
        self.segmentation_visibility_list.itemChanged.connect(self.toggle_volume_visibility)
        left_layout.addWidget(self.segmentation_visibility_list)
 
        # Save segmentations button
        self.save_all_btn = QPushButton("Save Updated Segmentations")
        self.save_all_btn.clicked.connect(self.save_modified_segmentations)
        self.save_all_btn.setEnabled(False)
        left_layout.addWidget(self.save_all_btn)

        # Create match labels button
        self.match_btn = QPushButton("Match Labels")
        self.match_btn.clicked.connect(lambda: self.toggle_label_matching())
        left_layout.addWidget(self.match_btn)

        self.layer_pick_btn = QPushButton("Pick Layer")
        self.layer_pick_btn.clicked.connect(self.toggle_layer_picking)
        left_layout.addWidget(self.layer_pick_btn)

        return left_panel

    def handle_click(self, obj, event):
        if self.picking_handler.active:
            self.picking_handler.on_click(obj, event)
        elif self.layer_picking_handler.active:
            self.layer_picking_handler.on_click(obj, event)

    def toggle_layer_picking(self):
        if self.layer_picking_handler.active:
            self.layer_picking_handler.deactivate()
            self.iren.GetInteractorStyle().DisablePicking()
            self.layer_pick_btn.setText("Pick Layer")
        else:
            self.picking_handler.deactivate()  # Ensure other picker is off
            self.layer_picking_handler.activate()
            self.iren.GetInteractorStyle().EnablePicking()
            self.layer_pick_btn.setText("Stop Layer Picking")


    def toggle_label_matching(self):
        if self.picking_handler.active:
            self.picking_handler.deactivate()
            self.iren.GetInteractorStyle().DisablePicking()
            self.match_btn.setText("Match Labels")
        else:
            self.picking_handler.activate()
            self.iren.GetInteractorStyle().EnablePicking()
            self.match_btn.setText("Stop Matching")

    def toggle_volume_visibility(self, item):
        idx = self.segmentation_visibility_list.row(item)
        is_visible = item.checkState() == Qt.CheckState.Checked
        self.segmentation_manager.get_segmentation_by_index(idx).visible = is_visible
        self.visualizer.update_visualization(False)

    def load_multiple_masks(self):
        files, _ = QFileDialog.getOpenFileNames(
                    self,
            "Select Mask Files",
            filter="NRRD mask files (*mask.nrrd)"
        )
        
        try:
            self.segmentation_manager.load_segmentations(files)
        finally:
            self.update_checkboxes()
            self.visualizer.update_visualization(reset_camera=True)

    def update_checkboxes(self): 
        self.segmentation_visibility_list.clear()
        for s in self.segmentation_manager.segmentations:
                item = QListWidgetItem(s.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if s.visible else Qt.CheckState.Unchecked)
                self.segmentation_visibility_list.addItem(item)

    def save_modified_segmentations(self):
        if not self.segmentation_manager.has_modification():
            print("No modifications to save")
            return
        
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Updated Segmentations"
        )

        self.segmentation_manager.save_updated_segmentations(Path(save_dir))

    def keypress_callback(self, obj, event):
        key = obj.GetKeySym().lower()
        if key == 'q':
            self.close_application()
        elif key == 'l':
            self.load_multiple_masks()
        elif key == 'm':
            self.toggle_label_matching()
        elif key == 'escape':
            print("Cancelled matching")
            self.picking_handler.deactivate()
            self.update_visualization()
        elif key == 'p':
            self.toggle_layer_picking()


    def close_application(self):
        self.close()
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
