from pathlib import Path
from dataclasses import dataclass
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QHBoxLayout, QWidget, QDialog, QLabel, QFileDialog, QListWidget, QListWidgetItem, QShortcut, QSizePolicy)
from PyQt5.QtGui import QKeySequence
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from PyQt5.QtCore import Qt
from nrrd_file import Nrrd
from vtkmodules.util.numpy_support import numpy_to_vtk
from match_stitches import match_stitches

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
        opacity_tf.AddPoint(1, 1.0)
        
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

class MainWindow(QMainWindow):
    segmentation_manager: SegmentationManager

    def __init__(self, parent=None, segmentation_manager = SegmentationManager()):
        super(MainWindow, self).__init__(parent)
        self.segmentation_manager = segmentation_manager
        
        # Matching state
        self.matching_state = None  # Can be None, 'selecting_giver', or 'selecting_receiver'
        self.giver_selected = None
        
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

        self.visualizer = SegmentationVisualizer(self.renderer, self.segmentation_manager)
        
        # Set up picker
        self.picker = vtk.vtkCellPicker()
        self.iren.SetPicker(self.picker)
        
        # Add click callback
        self.iren.AddObserver("LeftButtonPressEvent", self.on_click)
        
        # Set up keyboard interaction
        self.iren.AddObserver('KeyPressEvent', self.keypress_callback)
        
        self.setCentralWidget(self.central_widget)
        self.show()
        self.iren.Initialize()

        self.quit_shortcut = QShortcut(QKeySequence('q'), self)
        self.quit_shortcut.activated.connect(self.close_application)


    @property
    def next_seg_id(self) -> int:
        return self.segmentation_manager.number_of_segmentations() + 1


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
        self.match_btn.clicked.connect(self.start_matching)
        left_layout.addWidget(self.match_btn)

        return left_panel

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

    def start_matching(self):
        if self.segmentation_manager.number_of_segmentations() < 2:
            print("Need at least 2 segmentations to match labels")
            return
        
        self.matching_state = 'selecting_giver'
        print("Select giver volume (will be highlighted in green)")
        self.visualizer.update_visualization()

    def on_click(self, obj, event):
        if self.matching_state is None:
            return

        # Get click position
        click_pos = self.iren.GetEventPosition()
        
        # Perform pick
        if self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer):
            picked_volume = self.picker.GetProp3D()
            
            # Find which segmentation was picked among visible ones
            picked_seg = None
            for seg in self.segmentation_manager.segmentations:
                if seg.visible and seg.volume_actor == picked_volume:
                    picked_seg = seg
                    break
            
            if picked_seg:
                if self.matching_state == 'selecting_giver':
                    self.giver_selected = picked_seg
                    self.matching_state = 'selecting_receiver'
                    print("Now select receiver volume (will be highlighted in red)")
                elif self.matching_state == 'selecting_receiver':
                    if picked_seg == self.giver_selected:
                        print("Cannot select same volume as giver and receiver")
                        return
                    
                    try:
                        updated_seg = match_stitches(self.giver_selected.masks[-1], 
                                                   picked_seg.masks[-1])
                        picked_seg.masks.append(updated_seg)
                        self.save_all_btn.setEnabled(True)
                        print("Successfully matched labels")
                    except Exception as e:
                        print(f"Error matching labels: {str(e)}")

                    # Reset state
                    self.matching_state = None
                    self.giver_selected = None
                
                self.visualizer.update_visualization()

    def keypress_callback(self, obj, event):
        key = obj.GetKeySym().lower()
        if key == 'q':
            self.close_application()
        elif key == 'escape' and self.matching_state is not None:
            self.matching_state = None
            self.giver_selected = None
            print("Cancelled matching")
            self.update_visualization()

    def close_application(self):
        self.close()
        QApplication.quit()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

