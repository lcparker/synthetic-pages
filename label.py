from pathlib import Path
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                            QHBoxLayout, QWidget, QDialog, QLabel, QFileDialog, QListWidget, QListWidgetItem, QShortcut, QSizePolicy)
from PyQt5.QtGui import QKeySequence
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from PyQt5.QtCore import Qt
from nrrd_file import Nrrd
from vtkmodules.util.numpy_support import numpy_to_vtk
import numpy as np
from match_stitches import match_stitches

class LoadSegmentationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Segmentation")
        self.setModal(True)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Name section
        name_widget = QWidget()
        name_layout = QHBoxLayout(name_widget)
        self.name_label = QLabel("Name:")
        self.name_input = QLabel("Segmentation " + str(parent.next_seg_id))
        name_layout.addWidget(self.name_label)
        name_layout.addWidget(self.name_input)
        layout.addWidget(name_widget)
        
        # Volume section (optional)
        volume_widget = QWidget()
        volume_layout = QHBoxLayout(volume_widget)
        self.volume_label = QLabel("(Optional) No volume selected")
        volume_button = QPushButton("Select Volume")
        volume_button.setAutoDefault(False)
        volume_button.setDefault(False)
        volume_button.clicked.connect(self.select_volume)
        volume_layout.addWidget(self.volume_label)
        volume_layout.addWidget(volume_button)
        layout.addWidget(volume_widget)
        
        # Mask section
        mask_widget = QWidget()
        mask_layout = QHBoxLayout(mask_widget)
        self.mask_label = QLabel("No mask selected")
        mask_button = QPushButton("Select Mask")
        mask_button.setAutoDefault(False)
        mask_button.setDefault(False)
        mask_button.clicked.connect(self.select_mask)
        mask_layout.addWidget(self.mask_label)
        mask_layout.addWidget(mask_button)
        layout.addWidget(mask_widget)
        
        # Load button
        load_button = QPushButton("Load")
        load_button.clicked.connect(self.accept)
        layout.addWidget(load_button)
        
        self.volume_path = None
        self.mask_path = None

    def select_volume(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Volume File", filter="NRRD files (*.nrrd)")
        if path:
            self.volume_path = path
            self.volume_label.setText(path.split('/')[-1])

    def select_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Mask File", filter="NRRD files (*.nrrd)")
        if path:
            self.mask_path = path
            self.mask_label.setText(path.split('/')[-1])

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.next_seg_id = 1
        
        # Store loaded segmentations as list of dicts
        self.segmentations = []
        
        # Matching state
        self.matching_state = None  # Can be None, 'selecting_giver', or 'selecting_receiver'
        self.giver_selected = None
        
        # Create main layout
        self.central_widget = QWidget()
        main_layout = QHBoxLayout(self.central_widget)
        
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Create load segmentation button
        self.load_seg_btn = QPushButton("Load Segmentation")
        self.load_seg_btn.clicked.connect(self.show_load_dialog)
        left_layout.addWidget(self.load_seg_btn)

        self.load_multiple_btn = QPushButton("Load Multiple Masks")
        self.load_multiple_btn.clicked.connect(self.load_multiple_masks)
        left_layout.addWidget(self.load_multiple_btn)
        
        # Create list of loaded segmentations
        self.seg_list = QListWidget()
        self.seg_list.itemChanged.connect(lambda _: self.update_visualization(False))  # For checkbox changes
        left_layout.addWidget(self.seg_list)
 
        # Save segmentations button
        self.save_all_btn = QPushButton("Save Updated Segmentations")
        self.save_all_btn.clicked.connect(self.save_modified_segmentations)
        self.save_all_btn.setEnabled(False)
        left_layout.addWidget(self.save_all_btn)

        # Create match labels button
        self.match_btn = QPushButton("Match Labels")
        self.match_btn.clicked.connect(self.start_matching)
        left_layout.addWidget(self.match_btn)
        
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

        # Initialize color map
        self.color_map = None
        self.initialize_color_map()

        self.quit_shortcut = QShortcut(QKeySequence('q'), self)
        self.quit_shortcut.activated.connect(self.close_application)

    def close_application(self):
        self.close()
        QApplication.quit()

    def initialize_color_map(self):
        """Create a consistent color mapping using the curated color set"""
        self.color_map = vtk.vtkColorTransferFunction()
        
        # Make 0 (air) black/transparent
        self.color_map.AddRGBPoint(0, 0, 0, 0)
        
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
            
            self.color_map.AddRGBPoint(i, r, g, b)


    def load_multiple_masks(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Mask Files",
            filter="NRRD mask files (*mask.nrrd)"
        )
        
        if files:
            for mask_path in files:
                try:
                    # Parse coordinates from filename
                    filename = mask_path.split('/')[-1]
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
                    name = f"Seg ({z}, {y}, {x})"  # Changed order here
                    
                    # Construct corresponding volume path
                    volume_path = mask_path.replace('_mask.nrrd', '_volume.nrrd')
                    if not Path(volume_path).exists():
                        volume_path = None
                    
                    # Load the segmentation
                    self.load_segmentation(name, volume_path, mask_path)
                    
                except Exception as e:
                    print(f"Error loading {mask_path}: {str(e)}")


    def keypress_callback(self, obj, event):
        key = obj.GetKeySym().lower()
        if key == 'q':
            self.close_application()
        elif key == 'escape' and self.matching_state is not None:
            self.matching_state = None
            self.giver_selected = None
            print("Cancelled matching")
            self.update_visualization()
 
    def show_load_dialog(self):
        dialog = LoadSegmentationDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            if dialog.mask_path:  # Only check for mask path
                self.load_segmentation(dialog.name_input.text(), dialog.volume_path, dialog.mask_path)
            else:
                print("Please select a mask file")

    def load_segmentation(self, name, volume_path, mask_path):
        try:
            # Load nrrd files
            mask = Nrrd.from_file(mask_path)
            new_seg = {
                'name': name,
                'volume': Nrrd.from_file(volume_path) if volume_path else None,
                'mask_history': [mask],  # List of masks, first one is original
                'volume_actor': None,
                'visible': True
            }
            
            self.segmentations.append(new_seg)
 
            # Create list item with checkbox
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.seg_list.addItem(item)
            
            self.next_seg_id += 1
            
            print(f"Successfully loaded mask for {name}")
            self.update_visualization(reset_camera=True)
            
        except Exception as e:
            print(f"Error loading files: {str(e)}")

    def start_matching(self):
        if len(self.segmentations) < 2:
            print("Need at least 2 segmentations to match labels")
            return
        
        self.matching_state = 'selecting_giver'
        print("Select giver volume (will be highlighted in green)")
        self.update_visualization()

    def save_modified_segmentations(self):
        # Check if any segmentations have modifications
        has_modifications = any(len(seg['mask_history']) > 1 for seg in self.segmentations)
        
        if not has_modifications:
            print("No modifications to save")
            return
            
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Updated Segmentations"
        )
        
        if save_dir:
            try:
                save_dir = Path(save_dir)
                saved_count = 0
                
                for seg in self.segmentations:
                    if len(seg['mask_history']) > 1:  # Has modifications
                        # Extract coordinates from name
                        name = seg['name']
                        coords = name.replace("Seg (", "").replace(")", "").split(", ")
                        z, y, x = coords
                        
                        # Save most recent modification
                        new_filename = f"{z}_{y}_{x}_mask.nrrd"
                        save_path = save_dir / new_filename
                        seg['mask_history'][-1].write(str(save_path))
                        saved_count += 1
                
                print(f"Successfully saved {saved_count} modified segmentations to {save_dir}")
                    
            except Exception as e:
                print(f"Error saving segmentations: {str(e)}")


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
            for seg in self.segmentations:
                if seg['visible'] and seg.get('volume_actor') == picked_volume:
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
                        updated_seg = match_stitches(self.giver_selected['mask_history'][-1], 
                                                   picked_seg['mask_history'][-1])
                        picked_seg['mask_history'].append(updated_seg)
                        self.save_all_btn.setEnabled(True)
                        print("Successfully matched labels")
                    except Exception as e:
                        print(f"Error matching labels: {str(e)}")

                    # Reset state
                    self.matching_state = None
                    self.giver_selected = None
                
                self.update_visualization()

    def update_visualization(self, reset_camera=False):
        # Clear existing actors
        self.renderer.RemoveAllViewProps()
        
        # Update visibility flags from checkboxes
        for i in range(self.seg_list.count()):
            item = self.seg_list.item(i)
            self.segmentations[i]['visible'] = (item.checkState() == Qt.Checked)
        
        # Visualize all visible segmentations
        for seg in self.segmentations:
            if seg['visible']:
                self.visualize_segmentation(seg)
            
        # Only reset camera when explicitly requested
        if reset_camera:
            self.renderer.ResetCamera()
            
        self.vtk_widget.GetRenderWindow().Render()


    def visualize_segmentation(self, segmentation):
        # Get mask data and metadata
        # Convert from NRRD standard ordering (ZYX) to VTK's expected ordering (XYZ)
        mask_data = segmentation['mask_history'][-1].volume.T
        metadata = segmentation['mask_history'][-1].metadata
        
        # Convert numpy array to VTK image data
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(mask_data.shape)
        vtk_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        # Flatten the numpy array and convert to VTK array
        vtk_array = numpy_to_vtk(mask_data.flatten(), deep=True)
        vtk_data.GetPointData().GetScalars().DeepCopy(vtk_array)

        # Create color transfer function based on state
        if self.matching_state == 'selecting_giver':
            color_tf = vtk.vtkColorTransferFunction()
            for i in range(32):
                color_tf.AddRGBPoint(i, 0.7, 1.0, 0.7)  # Light green
        elif self.matching_state == 'selecting_receiver':
            color_tf = vtk.vtkColorTransferFunction()
            if segmentation == self.giver_selected:
                for i in range(32):
                    color_tf.AddRGBPoint(i, 0.0, 1.0, 0.0)  # Bright green
            else:
                for i in range(32):
                    color_tf.AddRGBPoint(i, 1.0, 0.7, 0.7)  # Light red
        else:
            color_tf = self.color_map
        
        # Create opacity transfer function
        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(0, 0.0)  # Air is transparent
        opacity_tf.AddPoint(1, 1.0)  # All other indices semi-transparent
        
        # Create volume property
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_tf)
        volume_property.SetScalarOpacity(opacity_tf)
        volume_property.ShadeOn()
        
        # Create volume mapper
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_data)
        
        # Create volume actor
        volume = vtk.vtkVolume()
        volume.SetMapper(mapper)
        volume.SetProperty(volume_property)
        
        # Set the position and orientation using metadata
        matrix = vtk.vtkMatrix4x4()
        
        # Set rotation from space directions
        space_directions = metadata['space directions']
        for i in range(3):
            for j in range(3):
                matrix.SetElement(i, j, space_directions[i][j])
        
        # Set translation from space origin
        space_origin = metadata['space origin']
        for i in range(3):
            matrix.SetElement(i, 3, space_origin[i])
        
        # Apply transformation to volume
        volume.SetUserMatrix(matrix)
        
        # Store volume actor reference
        segmentation['volume_actor'] = volume
        
        # Add to renderer
        self.renderer.AddVolume(volume)

    def hsv_to_rgb(self, h, s, v):
        if s == 0.0: return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0: return (v, t, p)
        if i == 1: return (q, v, p)
        if i == 2: return (p, v, t)
        if i == 3: return (p, q, v)
        if i == 4: return (t, p, v)
        if i == 5: return (v, p, q)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

