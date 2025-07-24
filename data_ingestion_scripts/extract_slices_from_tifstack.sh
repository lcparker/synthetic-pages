#!/bin/bash
# More robust version that handles potential issues

archive_file="$1"
start_num=${2:-3840}
end_num=${3:-4096}

# Check if 7zz exists
if ! command -v 7zz &> /dev/null; then
    echo "7zz not found. Please install 7-Zip."
    exit 1
fi

# Check if archive exists
if [[ ! -f "$archive_file" ]]; then
    echo "Archive file not found: $archive_file"
    exit 1
fi

# Extract files in range
for i in $(seq $start_num $end_num); do
    filename=$(printf "%05d.tif" $i)
    echo "Extracting: $filename"
    7zz e "$archive_file" -ir!"$filename" -y
done
