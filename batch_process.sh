#!/bin/bash
# chmod +x batch_process.sh
# Directory containing the images
IMAGE_DIR="/Volumes/Solardisk/sunplanet/040124/timelapse5sec200/VC"

# Output directory
OUTPUT_DIR="/Volumes/Solardisk/sunplanet/040124/timelapse5sec200/SF"

# Iterate over each image in the directory
for image in "$IMAGE_DIR"/*.tif; do
    echo "Processing $image..."
    python SolarFinish.py "$image" --brighten 0.62 --brightenweight 1.0 --enhance 1.8,3.85 --crop no --rotate 0.0 --darkclip 0.016 --colorize no -o "$OUTPUT_DIR"
done

echo "Processing complete."

#SolarFinish --brighten 0.62 --brightenweight 1.0 --enhance 1.8,3.85 --crop 1.4 --flip h --rotate 0.0 --darkclip 0.016 --colorize no
#--brighten 0.45 --brightenweight 0.5 --enhance 1.9,3.55 --crop no --rotate 0.0 --darkclip 0.016 --colorize no -o "$OUTPUT_DIR"
