#!/bin/bash

# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 START_NUM END_NUM"
    echo "Downloads scroll images from START_NUM to END_NUM"
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

for i in $(seq -f "%05g" $Z_START $Z_END); do
    wget "http://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes/20241024131838/${i}.tif"
done
