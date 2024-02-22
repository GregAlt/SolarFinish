#!/bin/bash
# chmod +x batch_process.sh
# Directory containing the images
IMAGE_DIR="/Volumes/Solardisk/sunplanet/230124-15sek23januar/ser-timelapses-15sek-400/pss"

# Output directory
OUTPUT_DIR="/Volumes/Solardisk/sunplanet/230124-15sek23januar/ser-timelapses-15sek-400/SF-toAI"

# Iterate over each image in the directory
for image in "$IMAGE_DIR"/*.tiff; do
    echo "Processing $image..."
    python SolarFinish.py "$image" --brighten 0.5 --brightenweight 1.0  --enhance 3.65,5.0 --crop 1.4 --rotate 0.0 --darkclip 0.015 --colorize no --silent -o "$OUTPUT_DIR"
done

echo "Processing complete."

#ffmpeg -r 24 -f image2 -i suni_%05d.png -c:v prores_ks -profile:v 2 -crf 25 -pix_fmt yuv420p solar_ekeberg_sf.mov

#python SolarFinish.py --brighten 0.7 --brightenweight 1.0  --deconvolution --enhance 1.7,4.3 --crop no --rotate 0.0 --darkclip 0.016 --colorize no
#--brighten 0.45 --brightenweight 0.5 --enhance 1.9,3.55 --crop no --rotate 0.0 --darkclip 0.016 --colorize no -o "$OUTPUT_DIR"
#python SolarFinish.py --brighten 0.7 --brightenweight 1.0  --enhance no --crop 1.4 --rotate 0.0 --darkclip 0.016 --colorize no