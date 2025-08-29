import jax
import xarray

import numpy as np
import haiku as hk
import jax.numpy as jnp


import graphcast.casting
import graphcast.samplers_utils
from graphcast import normalization
from graphcast import xarray_tree
from graphcast import xarray_jax
from graphcast.rollout import _get_next_inputs

from .data_loading import init_gcs, load_normalization_data, load_model

### GLOBAL STATE

GCS_BUCKET = init_gcs()
diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level = load_normalization_data(GCS_BUCKET)
construct_wrapped_gencast, params, state, task_config = load_model(GCS_BUCKET)

### GLOBAL STATE

@hk.transform_with_state
def exact_forward_fn(inputs, targets, forcings):
    predictor = construct_wrapped_gencast()
    denoised_predictions = predictor(
        inputs, targets, forcings
    )

    return denoised_predictions
forward_fn_jitted = jax.jit(
    lambda rng, i, t, f: exact_forward_fn.apply(params, state, rng, i, t, f)[0]
)

@hk.transform_with_state
def approx_forward_fn(inputs, targets, forcings, approximation_steps):
    nan_cleaner = construct_wrapped_gencast()
    normalizer = nan_cleaner._predictor
    predictor = normalizer._predictor

    if nan_cleaner._var_to_clean in inputs.keys():
        inputs = nan_cleaner._clean(inputs)
    if nan_cleaner._var_to_clean in targets.keys():
        targets = nan_cleaner._clean(targets)
    if forcings and nan_cleaner._var_to_clean in forcings.keys():
        forcings = nan_cleaner._clean(forcings)

    norm_inputs = normalization.normalize(inputs, normalizer._scales, normalizer._locations)
    norm_forcings = normalization.normalize(forcings, normalizer._scales, normalizer._locations)
    norm_target_residuals = xarray_tree.map_structure(
        lambda t: normalizer._subtract_input_and_normalize_target(inputs, t),
        targets
    )

    dtype = graphcast.casting.infer_floating_dtype(targets)  # pytype: disable=wrong-arg-types
    batch_size = inputs.sizes['batch']

    key = hk.next_rng_key()
    noise_levels = xarray_jax.DataArray(
        data=graphcast.samplers_utils.rho_inverse_cdf(
            min_value=predictor._noise_config.training_min_noise_level,
            max_value=predictor._noise_config.training_max_noise_level,
            rho=predictor._noise_config.training_noise_level_rho,
            cdf=jax.random.uniform(key, minval=(approximation_steps - 1)/approximation_steps, maxval=1, shape=(batch_size,), dtype=dtype)
        ),
        dims=('batch',)
    )
    noise = (
        graphcast.samplers_utils.spherical_white_noise_like(targets) * noise_levels
    )
    x_current = norm_target_residuals + noise

    for step in range(approximation_steps):
        # make actual predictions
        denoised_predictions = predictor._preconditioned_denoiser(
            norm_inputs, x_current, noise_levels, norm_forcings
        )
        if step == approximation_steps - 1:
            continue # skip rest, as no next sample exists

        new_noise_levels = xarray_jax.DataArray(
            data=graphcast.samplers_utils.rho_inverse_cdf(
                min_value=predictor._noise_config.training_min_noise_level,
                max_value=predictor._noise_config.training_max_noise_level,
                rho=predictor._noise_config.training_noise_level_rho,
                cdf=jax.random.uniform(
                    key,
                    minval=(approximation_steps - step - 2)/approximation_steps,
                    maxval=(approximation_steps - step - 1)/approximation_steps,
                    shape=(batch_size,),
                    dtype=dtype
                )
            ),
            dims=('batch',)
        )
        next_over_current = new_noise_levels / noise_levels
        x_next = (1 - next_over_current) * denoised_predictions + next_over_current * x_current
        
        noise_levels = new_noise_levels
        x_current = x_next

    denoised_predictions = xarray_tree.map_structure(
        lambda pred: normalizer._unnormalize_prediction_and_add_input(inputs, pred),
        denoised_predictions,
    )

    return denoised_predictions
approx_forward_fn_jitted = jax.jit(
    lambda rng, inputs, targets, forcings, approximation_steps: approx_forward_fn.apply(
        params, state, rng, inputs, targets, forcings, approximation_steps
    )[0],
    static_argnums=[4],
)

def multi_step_forward(rng, inputs, targets, forcings, forward_fn):
    forcings_coords = xarray_jax.unwrap_coords(forcings.isel(time=slice(0, 1)))
    targets_coords = xarray_jax.unwrap_coords(targets.isel(time=slice(0, 1)))
    forcings_dims = {k: forcings[k].dims for k in forcings.data_vars}
    targets_dims = {k: targets[k].dims for k in targets.data_vars}
    def body_fn(current_inputs, arrays):
        current_targets, current_forcings = arrays
        current_targets = xarray.Dataset(
            {
                k: (
                    targets_dims[k],
                    xarray_jax.wrap(
                        jnp.expand_dims(jnp.expand_dims(v, 0), 0)
                        )
                    )
                for k, v in current_targets.items()
            },
            targets_coords,
        )
        current_forcings = xarray.Dataset(
            {
                k: (forcings_dims[k], xarray_jax.wrap(jnp.expand_dims(v, [0, 1])))
                for k, v in current_forcings.items()
            },
            forcings_coords,
        )

        predictions = forward_fn(
            rng, current_inputs, current_targets, current_forcings,
        )
        next_frame = xarray.merge([predictions, current_forcings])
        next_inputs = _get_next_inputs(current_inputs, next_frame)
        
        next_inputs = next_inputs.assign_coords(time=current_inputs.coords["time"])
        return next_inputs, xarray_jax.unwrap_vars(predictions)
    
    arrays = (
        xarray_jax.unwrap_vars(targets.isel(batch=0)),
        xarray_jax.unwrap_vars(forcings.isel(batch=0)),
    )
    _, all_predictions = jax.lax.scan(jax.checkpoint(body_fn), inputs, arrays)
    all_predictions = xarray.Dataset(
        {
            k: (targets_dims[k], xarray_jax.wrap(jnp.expand_dims(jnp.squeeze(v), 0)))
            for k, v in all_predictions.items()
        },
        xarray_jax.unwrap_coords(targets)
    )
    return all_predictions
multi_step_forward_jit = jax.jit(multi_step_forward, static_argnums=4)

def build_static_data_selector(coords, lat_start, lat_end, lon_start, lon_end):
    def find_index(haystack, needle):
        return np.argmin(np.abs(haystack.data - needle))
    lat_start = find_index(coords["lat"], lat_start)
    lat_end = find_index(coords["lat"], lat_end)

    def convert_lon(lon):
        if lon < 0:
            return lon + 360
        return lon

    lon_start = find_index(coords["lon"], convert_lon(lon_start))
    lon_end = find_index(coords["lon"], convert_lon(lon_end))
    # add 1 to include last point
    lat_end += 1
    lon_end += 1

    def selector(data):
        data = jnp.roll(data, -lon_start, axis=-1)
        
        inner_lon_end = (lon_end - lon_start) % len(coords["lon"].data)
        return data[..., lat_start:lat_end, 0:inner_lon_end]
    return selector