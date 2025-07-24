# Voxelise the GP 2023 dataset
set -e

# Download the GP tifstack
wget https://dl.ash2txt.org/datasets/grand-prize-banner-region/volumes/gp_tifstack.7z

# Download the segment meshes
wget https://dl.ash2txt.org/datasets/grand-prize-banner-region/gp_meshes.7z
7zz x gp_meshes.7z

# Convert the meshes to ZYX
mkdir -p gp_meshes_zyx
python data_ingestion_scripts/xyz_mesh_to_zyx.py gp_meshes/*.obj --output-dir gp_meshes_zyx

# extract the desired range of slices from the tifstack
bash extract_tifstack_slices.sh gp_tifstack.7z 3840 4096
mkdir gp_tifstack/
mkdir gp_tifstack_uncompressed
mv *.tif gp_tifstack/
for file in `fd gp_tifstack/`; do
    tiffcp -c 0 "$file" "gp_tifstack_uncompressed/$(basename "$file")"
done

# Convert the tifstack to nrrd format and crop the meshes
python data_ingestion_scripts/tiffs-to-nrrds-with-labels-unrolled.py \
    --tifstack gp_tifstack.7z \
    --mesh-dir gp_meshes_zyx \
    --output-dir gp_nrrds \
    --start-slice 3840 \
    --end-slice 4096 \

# Generate instance labels from the output of the cropped meshes
python data_ingestion_scripts/instance_labels_from_cubes_and_meshes.py gp_nrrds