#!/bin/bash

# Check if parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU Parallel is not installed"
    echo "Install it using:"
    echo "  Ubuntu/Debian: sudo apt-get install parallel"
    echo "  MacOS: brew install parallel"
    exit 1
fi

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 START_NUM END_NUM"
    echo "Downloads scroll images in parallel from START_NUM to END_NUM"
    echo "Example: $0 1 10"
    exit 1
fi

Z_START=$1
Z_END=$2

# Validate input parameters are numbers
if ! [[ "$Z_START" =~ ^[0-9]+$ ]] || ! [[ "$Z_END" =~ ^[0-9]+$ ]]; then
    echo "Error: Both arguments must be numbers"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to download a single file
download_file() {
    num=$(printf "%05d" $1)
    url="http://dl.ash2txt.org//full-scrolls/Scroll3/PHerc332.volpkg/volumes/20231201141544/${num}.tif"
    
    # Add a small random delay (0-1 seconds) to further stagger requests
    sleep $(awk 'BEGIN{print rand()}')
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting download of ${num}.tif"
    if wget --no-verbose "$url" 2>> logs/errors.log; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Successfully downloaded ${num}.tif"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed to download ${num}.tif"
        return 1
    fi
}
export -f download_file

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting download of files ${Z_START} to ${Z_END}"
echo "Rate limiting: Minimum 4 seconds between requests per job"

# Generate sequence and run downloads in parallel with rate limiting and logging
seq $Z_START $Z_END | parallel --progress \
    --delay 4 \
    --joblog logs/parallel_job.log \
    --results logs/parallel_output \
    -j 4 download_file {}

# Print summary at the end
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download process completed"
echo "Logs available in:"
echo "- Error log: logs/errors.log"
echo "- Job log: logs/parallel_job.log"
echo "- Detailed output: logs/parallel_output/"
