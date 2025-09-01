#!/usr/bin/env bash

set -e

export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1

echo "** Step 1/3: Main evaluation: a,b,c (Total estimated time: 7.5h) **"
echo "** Step 1a/3: Wind (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --epsilons 0.03 0.05 --locations 10 --target=wind
echo "** Step 1b/3: Temperature (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --epsilons 0.03 0.05 --locations 10 --target=temperature
echo "** Step 1c/3: Precipitation (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --epsilons 0.03 0.05 --locations 10 --target=precipitation

echo "** Step 2/3: Case studies (Total estimated time: 30min) **"
echo "** Step 2a/3: Wind (Estimated time: 10min) **"
apptainer run --nv container.sif python3 src/case_study_heat.py
echo "** Step 2b/3: Temperature (Estimated time: 10min) **"
apptainer run --nv container.sif python3 src/case_study_heat.py
echo "** Step 2c/3: Precipitation (Estimated time: 10min) **"
apptainer run --nv container.sif python3 src/case_study_heat.py


echo "** Step 3/3: Generating report (Estimated time: 20s) **"
apptainer run container.sif python3 src/generate_report.py --short