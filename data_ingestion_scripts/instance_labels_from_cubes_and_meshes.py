#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List
import re
import logging

from synthetic_pages.instance_labels_from_meshes import generate_label_volume, load_page_meshes_from_zyx_header
from synthetic_pages.types.nrrd import Nrrd


# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Generate instance labels from mesh files and input volume NRRDs. Uses outputs of `tiffs-to-nrrds-with-labels-unrolled.py` for all position headers in a directory"
)

parser.add_argument(
    "directory",
    type=Path,
    help="Directory containing volume and mesh files"
)

parser.add_argument(
    "--papyrus-threshold",
    type=int,
    default=25248,
    help="Threshold value for papyrus detection (default: 25248)"
)

parser.add_argument(
    "--page-thickness",
    type=float,
    default=30.0,
    help="Page thickness in unitless coordinates (default: 30.0)"
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing label files (default: skip existing)"
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable verbose logging"
)

arguments = parser.parse_args()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG if arguments.verbose else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate directory
if not arguments.directory.exists():
    print(f"Error: Directory {arguments.directory} does not exist")
    exit(1)

if not arguments.directory.is_dir():
    print(f"Error: {arguments.directory} is not a directory")
    exit(1)

# Step 1: Group files by position header
logger.info("Step 1: Grouping files by position header...")

position_header_pattern = re.compile(r'^(\d{5}_\d{5}_\d{5})_')
file_groups: Dict[str, Dict[str, List[Path]]] = {}

for file_path in arguments.directory.iterdir():
    if not file_path.is_file():
        continue
        
    match = position_header_pattern.match(file_path.name)
    if match is None:
        logger.warning(f"Skipping file with unexpected name format: {file_path.name}")
        continue
    
    position_header = match.group(1)
    
    if position_header not in file_groups:
        file_groups[position_header] = {
            'volume': [],
            'meshes': [],
            'labels': [],
            'unexpected': []
        }
    
    filename = file_path.name
    if filename == f"{position_header}_volume.nrrd":
        file_groups[position_header]['volume'].append(file_path)
    elif filename == f"{position_header}_labels.nrrd":
        file_groups[position_header]['labels'].append(file_path)
    elif filename.startswith(f"{position_header}_volume_mesh_") and filename.endswith(".obj"):
        file_groups[position_header]['meshes'].append(file_path)
    else:
        file_groups[position_header]['unexpected'].append(file_path)

logger.info(f"Found {len(file_groups)} position header groups")

if not file_groups:
    logger.warning("No valid file groups found in directory")
    exit(0)

# Step 2: Validate each group has required files
logger.info("Step 2: Validating file groups...")

valid_groups = {}

for position_header, files in file_groups.items():
    # Must have exactly one volume file
    if len(files['volume']) == 0:
        logger.error(f"Skipping {position_header}: no volume file found")
        continue
    elif len(files['volume']) > 1:
        logger.error(f"Skipping {position_header}: multiple volume files found")
        continue
    
    # Warn if no meshes but don't skip
    if len(files['meshes']) == 0:
        logger.warning(f"No mesh files found for {position_header}")
    
    # Skip if unexpected files
    if files['unexpected']:
        unexpected_names = [f.name for f in files['unexpected']]
        logger.warning(f"Skipping {position_header}: unexpected files found: {unexpected_names}")
        continue
    
    valid_groups[position_header] = {
        'volume_file': files['volume'][0],
        'mesh_files': files['meshes']
    }
    
    logger.debug(f"Validated {position_header}: volume + {len(files['meshes'])} meshes")

logger.info(f"Validated {len(valid_groups)} groups for processing")

if not valid_groups:
    logger.warning("No valid groups remaining after validation")
    exit(0)

# Step 3: Process each group
logger.info("Step 3: Processing groups...")

successful_count = 0
failed_count = 0

for position_header in sorted(valid_groups.keys()):
    group = valid_groups[position_header]
    volume_file = group['volume_file']
    mesh_files = group['mesh_files']
    
    expected_labels_path = arguments.directory / f"{position_header}_labels.nrrd"
    
    # Skip if labels already exist (unless overwrite is specified)
    if not arguments.overwrite and expected_labels_path.exists():
        logger.info(f"Skipping {position_header} - labels already exist")
        successful_count += 1
        continue
    
    try:
        logger.info(f"Processing {position_header}")
        
        # Load volume
        input_volume = Nrrd.from_file(volume_file)
        logger.debug(f"Loaded volume from {volume_file}")
        
        # Load meshes (if any)
        if mesh_files:
            meshes_zyx = load_page_meshes_from_zyx_header(
                position_header, 
                arguments.directory
            )
            logger.debug(f"Loaded {len(meshes_zyx)} meshes")
        else:
            meshes_zyx = []
            logger.warning(f"No meshes available for {position_header}")
        
        # Generate labels
        labels = generate_label_volume(
            input_volume, 
            meshes_zyx,
            papyrus_threshold=arguments.papyrus_threshold,
            page_thickness_unitless=arguments.page_thickness
        )
        
        # Save labels
        labels.write(expected_labels_path)
        logger.info(f"Generated labels: {expected_labels_path}")
        
        successful_count += 1
        
    except Exception as error:
        logger.error(f"Failed to process {position_header}: {error}")
        failed_count += 1

# Final summary
logger.info(f"Processing complete: {successful_count} successful, {failed_count} failed")