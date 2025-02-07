import os
import slicer

def cleanup_temp_nodes():
    """
    Clean up any leftover temporary labelmap nodes
    """
    # Find all labelmap nodes
    labelmap_nodes = slicer.util.getNodesByClass('vtkMRMLLabelMapVolumeNode')
    for node in labelmap_nodes:
        if node.GetName().startswith('LabelMapVolume'):
            slicer.mrmlScene.RemoveNode(node)

def load_nrrd_as_segmentation(nrrd_path):
    """
    Load an NRRD labelmap file as a segmentation in 3D Slicer
    
    Parameters:
    nrrd_path (str): Path to the NRRD file
    
    Returns:
    vtkMRMLSegmentationNode: The created segmentation node
    """
    # Create a segmentation node with name based on the file
    base_name = os.path.splitext(os.path.basename(nrrd_path))[0]
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', base_name)
    segmentation_node.CreateDefaultDisplayNodes()
    
    try:
        labelmap_node = slicer.util.loadLabelVolume(nrrd_path, properties={'labelmap': True})
        
        sucess = slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_node, segmentation_node)
            
    finally:
        # Always clean up the temporary labelmap node
        if labelmap_node:
            slicer.mrmlScene.RemoveNode(labelmap_node)
    
    return segmentation_node

def process_directory_nrrds(directory_path):
    """
    Load all NRRD files in the specified directory as segmentations
    
    Parameters:
    directory_path (str): Path to the directory containing NRRD files
    
    Returns:
    dict: Dictionary mapping filenames to their segmentation nodes
    """
    # First, clean up any existing temporary nodes
    cleanup_temp_nodes()
    
    # Normalize path and check if directory exists
    directory_path = os.path.normpath(directory_path)
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return {}
    
    # Dictionary to store the created segmentation nodes
    segmentation_nodes = {}
    
    # Process all NRRD files in the directory
    nrrd_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.nrrd')]
    for filename in nrrd_files:
        file_path = os.path.join(directory_path, filename)
        print(f"Processing: {filename}")
        
        # Load the segmentation
        segmentation_node = load_nrrd_as_segmentation(file_path)
        
        if segmentation_node:
            segmentation_nodes[filename] = segmentation_node
            print(f"Successfully loaded: {filename}")
        else:
            print(f"Failed to process: {filename}")
    
    # Final cleanup of any remaining temporary nodes
    cleanup_temp_nodes()
    
    # Print summary
    print(f"\nProcessing complete:")
    print(f"Total files processed: {len(nrrd_files)}")
    print(f"Successful loads: {len(segmentation_nodes)}")
    print(f"Failed loads: {len(nrrd_files) - len(segmentation_nodes)}")
    
    return segmentation_nodes

def create_colormap(max_id=32, colormap_name=None):  # colormap_name kept for compatibility
    """
    Create a colormap for segments using a curated set of visually distinct colors
    
    Parameters:
    max_id: int, maximum segment ID number to create colors for
    colormap_name: str, ignored (kept for compatibility)
    
    Returns:
    Dictionary mapping segment IDs to RGB colors as (r,g,b) tuples
    """
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
    
    # Create color dictionary
    colors = {}
    num_base_colors = len(base_colors)
    
    for i in range(max_id + 1):
        if i < num_base_colors:
            # Use base colors for first set of segments
            colors[i] = base_colors[i]
        else:
            # For additional segments, create variations of base colors
            base_idx = i % num_base_colors
            variation = (i // num_base_colors) + 1
            
            # Create a slightly different shade of the base color
            r, g, b = base_colors[base_idx]
            # Adjust brightness and saturation based on variation
            factor = 1.0 / (1.0 + variation * 0.3)
            colors[i] = (
                max(0.1, min(1.0, r * factor)),
                max(0.1, min(1.0, g * factor)),
                max(0.1, min(1.0, b * factor))
            )
    
    return colors

def show_segmentation_3d(segmentation_node, closed_surface=True, smoothing_factor=0.5, colormap_name='jet'):
    """
    Convert a segmentation to 3D mesh visualization
    
    Parameters:
    segmentation_node: vtkMRMLSegmentationNode to visualize
    closed_surface: bool, whether to create a closed surface representation
    smoothing_factor: float 0-1, amount of smoothing to apply
    """
    # Make sure we have a valid segmentation node
    if not segmentation_node:
        print("No segmentation node provided")
        return
        
    # Get segmentation display node
    display_node = segmentation_node.GetDisplayNode()
    if not display_node:
        print("Creating display node")
        segmentation_node.CreateDefaultDisplayNodes()
        display_node = segmentation_node.GetDisplayNode()
    
    # Make sure the closed surface representation exists
    if closed_surface and not segmentation_node.CreateClosedSurfaceRepresentation():
        print("Failed to create closed surface representation")
        return
        
    # Set visualization parameters
    display_node.SetVisibility(True)  # Make sure it's visible
    display_node.SetAllSegmentsVisibility(True)  # Show all segments
    
    # Set 3D view parameters
    display_node.SetPreferredDisplayRepresentationName3D("Closed surface")
    
    segmentation = segmentation_node.GetSegmentation()
    n_segments = segmentation.GetNumberOfSegments()
    colors = create_colormap()
    # Apply colors and settings to each segment
    for segment_index in range(n_segments):
        segment_id = segmentation.GetNthSegmentID(segment_index)
        segment = segmentation.GetSegment(segment_id)
        # Set color for this segment
        r, g, b = colors[int(segment_id)]
        segment.SetColor(r, g, b)
        
        display_node.SetSegmentOpacity3D(segment_id, 1.0)  # Full opacity
        # Set smoothing factor (0 = no smoothing, 1 = maximum smoothing)
        if hasattr(display_node, 'SetSegmentSmoothingFactor'):
            display_node.SetSegmentSmoothingFactor(segment_id, smoothing_factor)

def show_all_segmentations_3d(smoothing_factor=0.5):
    """
    Convert all segmentations in the scene to 3D mesh visualizations
    
    Parameters:
    smoothing_factor: float 0-1, amount of smoothing to apply
    """
    # Get all segmentation nodes in the scene
    segmentation_nodes = slicer.util.getNodesByClass('vtkMRMLSegmentationNode')
    
    if not segmentation_nodes:
        print("No segmentation nodes found in the scene")
        return
        
    print(f"Converting {len(segmentation_nodes)} segmentation(s) to 3D...")
    
    # Process each segmentation
    for node in segmentation_nodes:
        print(f"Processing segmentation: {node.GetName()}")
        show_segmentation_3d(node, smoothing_factor=smoothing_factor)
    
    print("Conversion complete!")


def display_segmentations_from_folder(directory: str, max_num_segmentations=20):
    directory = str(directory)
    segmentation_nodes = process_directory_nrrds(directory)
    for i, (_, node) in enumerate(segmentation_nodes.items()):
        if i > max_num_segmentations:
            return
        else:
            show_segmentation_3d(node)
