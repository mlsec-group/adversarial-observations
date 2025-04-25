from pathlib import Path

import jax
import xarray

import numpy as np
import jax.numpy as jnp

from graphcast import xarray_tree
from graphcast import xarray_jax
from graphcast import normalization


VARS_TO_ATTACK = [
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'geopotential',
    'mean_sea_level_pressure',
    'sea_surface_temperature',
    'specific_humidity',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
]

SCALES_PATH = Path(__file__).parent / "data" / "estimated_error_scales.nc"
with open(SCALES_PATH, "rb") as f:
    STDDEVS = xarray.load_dataset(f).compute()

def add_perturbation(inputs, perturbation):
    return inputs + normalization.unnormalize(perturbation, STDDEVS, None)


def scale_std(data):
    data = data - jnp.mean(data, axis=(-1, -2))[..., np.newaxis, np.newaxis]
    current_std = jnp.std(data, axis=(-1, -2))
    return data / current_std[..., np.newaxis, np.newaxis]


def projection(data, epsilon):
    data = xarray_jax.unwrap(data)
    data = data - jnp.mean(data, axis=(-1, -2))[..., np.newaxis, np.newaxis]
    current_std = jnp.std(data, axis=(-1, -2))
    current_std = current_std[..., np.newaxis, np.newaxis]
    data = data / current_std * jnp.minimum(current_std, epsilon)
    return xarray_jax.wrap(data)


def advdm(inputs, targets, forcings, epsilon, grads_fn, maxiter=10):
    # refer to Algorithm 1 of https://proceedings.mlr.press/v202/liang23g/liang23g.pdf
    # changed to 'l2' norm instead of l-inf norm
    # changes sign of grad to normalize by std
    # results in maximum std of perturbation <= epsilon
    alpha = epsilon / maxiter
    
    # zero init
    perturbation = xarray_tree.map_structure(lambda a: 0*a, inputs)
    for t in range(maxiter):
        perturbed_inputs = add_perturbation(inputs, perturbation)
        loss, grads = grads_fn(
            rng=jax.random.PRNGKey(t),
            inputs=perturbed_inputs,
            targets=targets,
            forcings=forcings,
            approximation_steps=1,
        )
        for var in VARS_TO_ATTACK:
            x = xarray_jax.unwrap_data(perturbation[var])
            diff = xarray_jax.unwrap_data(grads[var])
            diff = scale_std(diff) # scale to std = 1
            new_x = x - alpha * diff
            perturbation[var].data = xarray_jax.wrap(new_x)
        # logger.info(f"{t} {loss}")

    return perturbation


def dp_attacker(inputs, targets, forcings, epsilon, grads_fn, maxiter=10):
    # refer to Algorithm 1 of https://arxiv.org/abs/2405.19424
    # changed to 'l2' norm instead of l-inf norm
    # changes sign of grad to normalize by std
    alpha = 2 * epsilon / maxiter
    
    # zero init
    perturbation = xarray_tree.map_structure(lambda a: 0*a, inputs)
    for t in range(maxiter):
        perturbed_inputs = add_perturbation(inputs, perturbation)
        loss, grads = grads_fn(
            rng=jax.random.PRNGKey(t),
            inputs=perturbed_inputs,
            targets=targets,
            forcings=forcings,
            approximation_steps=1,
        )
        for var in VARS_TO_ATTACK:
            x = xarray_jax.unwrap_data(perturbation[var])
            diff = xarray_jax.unwrap_data(grads[var])
            diff = scale_std(diff) # scale to std = 1
            new_x = x - alpha * diff
            new_x = projection(new_x, epsilon)
            perturbation[var].data = xarray_jax.wrap(new_x)
        # logger.info(f"{t} {loss}")

    return perturbation


def our_attack(
        inputs,
        targets,
        forcings,
        epsilon,
        grads_fn,
        maxiter=10,
        do_log=False,
    ):
    beta = 0.9
    def _cos_anneal(eta_0, eta_min, t):
        return eta_min + 0.5 * (eta_0 - eta_min) * (1 + np.cos(t * np.pi / maxiter))
    _learning_rate = lambda t: _cos_anneal(2*epsilon, epsilon/maxiter, t)

    # zero init
    perturbation = xarray_tree.map_structure(lambda a: 0*a, inputs)
    first_moment = xarray_tree.map_structure(lambda a: 0*a, inputs)

    for t in range(maxiter):
        perturbed_inputs = add_perturbation(inputs, perturbation)
        loss, grads = grads_fn(
            rng=jax.random.PRNGKey(t),
            inputs=perturbed_inputs,
            targets=targets,
            forcings=forcings,
            approximation_steps=2,
        )

        for var in VARS_TO_ATTACK:
            x = xarray_jax.unwrap_data(perturbation[var])
            diff = xarray_jax.unwrap_data(grads[var])
            diff = scale_std(diff) # scale to std = 1
            diff = beta * xarray_jax.unwrap_data(first_moment[var]) + (1 - beta) * diff
            first_moment[var].data = xarray_jax.wrap(diff)
            learning_rate = _learning_rate(t) / (1 - beta**(t+1))
            new_x = x - learning_rate * diff
            new_x = projection(new_x, epsilon)
            perturbation[var].data = xarray_jax.wrap(new_x)
        if do_log:
            print(t, loss)

    return perturbation

def our_attack_wo_approximation(
        inputs,
        targets,
        forcings,
        epsilon,
        grads_fn,
        maxiter=10,
        do_log=False,
    ):
    beta = 0.9
    def _cos_anneal(eta_0, eta_min, t):
        return eta_min + 0.5 * (eta_0 - eta_min) * (1 + np.cos(t * np.pi / maxiter))
    _learning_rate = lambda t: _cos_anneal(2*epsilon, epsilon/maxiter, t)

    # zero init
    perturbation = xarray_tree.map_structure(lambda a: 0*a, inputs)
    first_moment = xarray_tree.map_structure(lambda a: 0*a, inputs)

    for t in range(maxiter):
        perturbed_inputs = add_perturbation(inputs, perturbation)
        loss, grads = grads_fn(
            rng=jax.random.PRNGKey(t),
            inputs=perturbed_inputs,
            targets=targets,
            forcings=forcings,
            approximation_steps=1,
        )

        for var in VARS_TO_ATTACK:
            x = xarray_jax.unwrap_data(perturbation[var])
            diff = xarray_jax.unwrap_data(grads[var])
            diff = scale_std(diff) # scale to std = 1
            diff = beta * xarray_jax.unwrap_data(first_moment[var]) + (1 - beta) * diff
            first_moment[var].data = xarray_jax.wrap(diff)
            learning_rate = _learning_rate(t) / (1 - beta**(t+1))
            new_x = x - learning_rate * diff
            new_x = projection(new_x, epsilon)
            perturbation[var].data = xarray_jax.wrap(new_x)
        if do_log:
            print(t, loss)

    return perturbation

def our_attack_wo_steps(
        inputs,
        targets,
        forcings,
        epsilon,
        grads_fn,
        maxiter=10,
        do_log=False,
    ):
    learning_rate = 2 * epsilon / maxiter
    # zero init
    perturbation = xarray_tree.map_structure(lambda a: 0*a, inputs)

    for t in range(maxiter):
        perturbed_inputs = add_perturbation(inputs, perturbation)
        loss, grads = grads_fn(
            rng=jax.random.PRNGKey(t),
            inputs=perturbed_inputs,
            targets=targets,
            forcings=forcings,
            approximation_steps=2,
        )

        for var in VARS_TO_ATTACK:
            x = xarray_jax.unwrap_data(perturbation[var])
            diff = xarray_jax.unwrap_data(grads[var])
            diff = scale_std(diff) # scale to std = 1
            new_x = x - learning_rate * diff
            new_x = projection(new_x, epsilon)
            perturbation[var].data = xarray_jax.wrap(new_x)
        if do_log:
            print(t, loss)

    return perturbation