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

 
def show_segmentation_3d(segmentation_node, closed_surface=True, smoothing_factor=0.5):
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
    
    # Apply smoothing
    segmentation = segmentation_node.GetSegmentation()
    for segment_index in range(segmentation.GetNumberOfSegments()):
        segment_id = segmentation.GetNthSegmentID(segment_index)
        display_node.SetSegmentOpacity3D(segment_id, 1.0)  # Full opacity
        # Set smoothing factor (0 = no smoothing, 1 = maximum smoothing)
        if hasattr(display_node, 'SetSegmentSmoothingFactor'):
            display_node.SetSegmentSmoothingFactor(segment_id, smoothing_factor)
    
    # Get/Create a 3D view
    layout_manager = slicer.app.layoutManager()
    layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)  # Switch to 3D view
    view_node = layout_manager.threeDWidget(0).mrmlViewNode()
    
    # Center the 3D view on the segmentation
    slicer.util.resetThreeDViews()

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

