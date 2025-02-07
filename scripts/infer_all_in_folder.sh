#!/bin/bash

# Print usage information
usage() {
    echo "Usage: $0 <directory> <weights>"
    echo "Example: $0 ./nrrds weights.pth"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    usage
fi

directory=$1
weights=$2

# Check if directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' does not exist"
    usage
fi

# Check if weights file exists
if [ ! -f "$weights" ]; then
    echo "Error: Weights file '$weights' does not exist"
    usage
fi

# Loop through all .nrrd files in specified directory
for input_file in "$directory"/*_volume.nrrd; do
    # Check if files exist (in case no matches were found)
    if [ ! -f "$input_file" ]; then
        echo "Error: No volume files found in $directory"
        exit 1
    fi
    
    # Create output filename by replacing 'volume' with 'mask'
    output_file="${input_file/volume/mask}"
    
    # Run inference.py with positional arguments: input output weights
    python synthetic-pages/inference.py "$input_file" "$output_file" "$weights"
    
    echo "Processed: $input_file -> $output_file"
done
