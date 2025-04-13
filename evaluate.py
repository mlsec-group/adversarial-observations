import os
import json
import argparse
import functools

import jax
import xarray

import numpy as np
import jax.numpy as jnp

from graphcast import xarray_jax

from data_loading import load_data
from model_running import (
    multi_step_forward_jit, forward_fn_jitted, build_static_data_selector,
    approx_forward_fn_jitted, task_config,
)
from attacks import add_perturbation, advdm, our_attack, dp_attacker

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


def select_precipitation(predictions):
    return predictions["total_precipitation_12hr"]

def select_temperature(predictions):
    return predictions["2m_temperature"]

def select_wind_speed(predictions):
    return np.sqrt(
        np.square(predictions["10m_u_component_of_wind"]) +
        np.square(predictions["10m_v_component_of_wind"])
    )
TARGET_SELECTORS = {
    "wind": select_wind_speed,
    "temperature": select_temperature,
    "precipitation": select_precipitation,
}

def general_loss_fn(
        rng,
        inputs,
        targets,
        forcings,
        forward_fn,
        variable_selection_fn,
        region_selection_fn,
    ):
    denoised_predictions = multi_step_forward_jit(
        rng,
        inputs,
        targets,
        forcings,
        forward_fn
    ).isel(time=-1)

    # select target variable
    target = variable_selection_fn(denoised_predictions)

    # select target region
    target = xarray_jax.unwrap_data(target, require_jax=True)
    target = region_selection_fn(target)

    # minus so that minimization of loss maximizes wind speed
    loss = -jnp.max(target)
    return loss

# no jit, as it does not provide a speedup in this case
precise_loss_fn = lambda rng, i, t, f, s, v: general_loss_fn(rng, i, t, f, forward_fn_jitted, v, s)

def adv_grads_fn(rng, inputs, targets, forcings, approximation_steps, variable_selection_fn, region_selection_fn):
    forward_fn = functools.partial(approx_forward_fn_jitted, approximation_steps=approximation_steps)
    def _aux(rng, i, t, f):
        loss = general_loss_fn(
            rng, i, t, f,
            forward_fn=forward_fn,
            variable_selection_fn=variable_selection_fn,
            region_selection_fn=region_selection_fn,
        )
        return loss

    loss, grads = jax.value_and_grad(
        _aux,
        argnums=1,
    )(rng, inputs, targets, forcings)
    return loss, grads
adv_grads_fn_jitted = jax.jit(adv_grads_fn, static_argnums=(4,5,6))


def path_for(epsilon, target, steps):
    return f"data/results/{target}_{epsilon}eps_{steps}steps.json"

def dump_results(results, epsilon, target, steps):
    with open(path_for(epsilon, target, steps), "w") as f:
        json.dump(results[epsilon], f)

def load_results(epsilon, target, steps):
    path = path_for(epsilon, target, steps)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        results = json.load(f)
    # ensure that all have the same length, otherwise the caching logic breaks
    shortest_length = min(len(v) for v in results.values())
    for k in results.keys():
        results[k] = results[k][:shortest_length]
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilons", type=float, nargs='+', required=True)
    parser.add_argument("--target", choices=TARGET_SELECTORS.keys(), required=True)
    parser.add_argument("--steps", type=int, default=50)

    args = parser.parse_args()


    with open("data/weather_evaluation_targets.json", "r") as f:
        targets = json.load(f)
    
    # era5 = xarray.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
    # cached variant:
    era5 = xarray.open_dataset("data/one_year_era5.nc")
    
    ATTACKS = {
        "Ours": our_attack,
        "AdvDM": advdm,
        "DP-Attacker": dp_attacker,
    }
    SEEDS = [1234567890, 123456789, 1234567, 123456, 12345]

    variable_selector = TARGET_SELECTORS[args.target]
    
    results = {}
    for epsilon in args.epsilons:
        # warm restart: test whether we already have data
        cached = load_results(epsilon, args.target, args.steps)
        if cached is not None:
            results[epsilon] = cached
            continue
        # initialize normally
        results[epsilon] = {k: [] for k in ATTACKS.keys()}
        results[epsilon]["unperturbed"] = []
    


    def test_across_seeds(*args):
        results = []
        for seed in SEEDS:
            loss = precise_loss_fn(
                jax.random.PRNGKey(seed),
                *args,
                variable_selector,
            )
            results.append(float(loss))
        return results

    for i, target in enumerate(targets):
        if all(len(results[epsilon]["unperturbed"]) > i for epsilon in args.epsilons):
            # assumming if unperturbed has it, all attacks also have it
            # is ensured during loading above and during saving below
            logger.info(f"Skipping sample {i+1}, already computed")
            continue
        logger.info(f"Loading sample {i+1}...")
        start_time = target["datetime"]
        inputs, targets, forcings = load_data(era5, start_time, task_config)
        logger.info("Successfully loaded data")

        # select appropriate target
        lat, lon = target["location"]["latitude"], target["location"]["longitude"]
        region_selection_fn = build_static_data_selector(inputs.coords, lat, lat, lon, lon)
        grads_fn = functools.partial(
            adv_grads_fn_jitted,
            region_selection_fn=region_selection_fn,
            variable_selection_fn=variable_selector
        )
        
        logger.info("Making initial prediction")
        predicted_targets = multi_step_forward_jit(
            jax.random.PRNGKey(0),
            inputs,
            targets,
            forcings,
            forward_fn_jitted,
        )
        
        baseline_results = test_across_seeds(
            inputs,
            targets,
            forcings,
            region_selection_fn,
        )
        logger.info(f"Baseline loss: {baseline_results}")
        for epsilon in args.epsilons:
            if len(results[epsilon]["unperturbed"]) > i:
                continue
            results[epsilon]["unperturbed"].append(baseline_results)

        for epsilon in args.epsilons:
            for name, fn in ATTACKS.items():
                if len(results[epsilon][name]) > i:
                    logger.info(f"Skipping sample {i+1} attack '{name}', already computed")
                    continue
                logger.info(f"Attack {name}")
                perturbation = fn(
                    inputs,
                    predicted_targets,
                    forcings,
                    epsilon,
                    grads_fn,
                    maxiter=args.steps,
                )
                perturbed_results = test_across_seeds(
                    add_perturbation(inputs, perturbation),
                    targets,
                    forcings,
                    region_selection_fn,
                )
                logger.info(f"{name} loss: {perturbed_results}")
                results[epsilon][name].append(perturbed_results)
            
            dump_results(results, epsilon, args.target, args.steps)
    
    for epsilon in args.epsilons:
        dump_results(results, epsilon, args.target, args.steps)