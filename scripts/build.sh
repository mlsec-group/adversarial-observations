#!/usr/bin/env bash

set -e

echo "** STEP 1/4: Build Apptainer Container (Estimated time: 10 min) **"
apptainer build container.sif container.def

echo "** STEP 2/4: Download geoBoundaries data (Estimated time: 5 min) **"
apptainer run containser.sif wget https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM0.geojson -O data/geoBoundariesCGAZ_ADM0.geojson

echo "** STEP 3/4: Download ERA5 dataset (Estimated time: 10 min) **"
apptainer run container.sif python3 src/cache_era5.py

echo "** STEP 4/4: Estimate variance of background error (Estimated time: 4 h) **"
apptainer run --nv container.sif python3 src/estimate_error_variance.py