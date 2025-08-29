#!/usr/bin/env bash

set -e

export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1

echo "** Step 1/4: Main evaluation: a,b,c (Total estimated time: 7.5h) **"
echo "** Step 1a/4: Wind (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --epsilons 0.03 0.05 --locations 10 --target=wind
echo "** Step 1b/4: Temperature (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --epsilons 0.03 0.05 --locations 10 --target=temperature
echo "** Step 1c/4: Precipitation (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --epsilons 0.03 0.05 --locations 10 --target=precipitation

echo "** Step 2/4: Ablation study: a,b,c (Total estimated time: 7.5h) **"
echo "** Step 2a/4: Wind (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --do-ablation --epsilons 0.03 0.05 --locations 10 --target=wind
echo "** Step 2b/4: Temperature (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --do-ablation --epsilons 0.03 0.05 --locations 10 --target=temperature
echo "** Step 2c/4: Precipitation (Estimated time: 2.5h) **"
apptainer run --nv container.sif python3 src/evaluate.py --do-ablation --epsilons 0.03 0.05 --locations 10 --target=precipitation

echo "** Step 3/4: Case studies (Total estimated time: 30min) **"
echo "** Step 3a/4: Wind (Estimated time: 10min) **"
apptainer run --nv container.sif python3 src/case_study_heat.py
echo "** Step 3b/4: Temperature (Estimated time: 10min) **"
apptainer run --nv container.sif python3 src/case_study_heat.py
echo "** Step 3c/4: Precipitation (Estimated time: 10min) **"
apptainer run --nv container.sif python3 src/case_study_heat.py


echo "** Step 4/4: Generating report (Estimated time: 20s) **"
apptainer run container.sif python3 src/generate_report.py --short