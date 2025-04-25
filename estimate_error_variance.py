from datetime import datetime, timedelta

import jax
import xarray

from tqdm import tqdm

import numpy as np
import jax.numpy as jnp

from graphcast import xarray_jax

from data_loading import load_data
from model_running import (
    forward_fn_jitted, task_config,
)

# Warning appears during jit-compilation of forward computation
# can be ignored, because while it is inefficient, it is only done twice
# during jit and the results are re-used afterwards
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)



if __name__ == "__main__":
    
    # era5 = xarray.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
    # cached variant:
    era5 = xarray.open_dataset("data/one_year_era5.nc")
    
    SEEDS = [1234567890, 123456789, 1234567, 123456, 12345]
    
    
    scales = {}
    for i in tqdm(range(365*2)):
        start_time = datetime.fromisoformat("2022-01-01T06:00:00") + timedelta(days=i/2)
        inputs, targets, forcings = load_data(era5, start_time.strftime("%Y-%m-%dT%H:%M:%S"), task_config, lead_time=timedelta(days=1))

        for seed in SEEDS:
            predicted_targets = forward_fn_jitted(
                jax.random.PRNGKey(seed),
                inputs,
                targets,
                forcings,
            )
            error = predicted_targets - targets
            vars = set(task_config.input_variables) & set(task_config.target_variables)
            
            for var in vars:
                values = xarray_jax.unwrap_data(error[var])
                if len(values.shape) > 0:
                    prediction_variance = jnp.std(values, axis=(-1, -2))
                    prediction_variance = prediction_variance.reshape(-1)
                    if var not in scales:
                        scales[var] = []
                    scales[var].append(prediction_variance)
                    continue
                prediction_variance = jnp.std(values[~jnp.isnan(values)])
                if var not in scales:
                    scales[var] = []
                scales[var].append(prediction_variance)
    
    # aggregate all scales
    mean_scales = {}
    for var in vars:
        values = jnp.vstack(scales[var])
        # mean of variance instead of mean of std
        mean_values = jnp.sqrt(jnp.mean(jnp.square(values), axis=0))
        mean_values = jnp.squeeze(mean_values)
        if len(mean_values.shape) > 0:
            mean_scales[var] = xarray.DataArray(mean_values, coords=dict(level=targets[var].coords["level"]))
        else:
            mean_scales[var] = xarray.DataArray(mean_values)
    mean_scales = xarray.Dataset(
        mean_scales 
    )
    mean_scales.to_netcdf("data/estimated_error_scales.nc")