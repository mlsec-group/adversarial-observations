#!/usr/bin/env bash

set -e

export JAX_CAPTURED_CONSTANTS_WARN_BYTES=-1

apptainer run --nv container.sif python3 src/evaluate.py --epsilons 0.02 --locations=1 --target=wind --steps=1

RESULT_PATH="data/results/wind_0.02eps_1steps.json"
if [ -f $RESULT_PATH ]; then
    echo "Everything seems to be fine."
    rm -f $RESULT_PATH
else
    echo "Error, expected results to exist after execution of the test"
fi