import json
import functools

from datetime import timedelta

import jax
import xarray
import matplotlib

import numpy as np
import jax.numpy as jnp

from graphcast import xarray_jax

from utils.data_loading import load_data
from utils.model_running import (
    multi_step_forward_jit, forward_fn_jitted, build_static_data_selector,
    approx_forward_fn_jitted, task_config,
)
from utils.attacks import add_perturbation, our_attack

# Warning appears during jit-compilation of forward computation
# can be ignored, because while it is inefficient, it is only done twice
# during jit and the results are re-used afterwards
import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore',SparseEfficiencyWarning)


def smooth(points):
    new_points = [points[0]]
    def distance(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    for point in points[1:]:
        if distance(point, new_points[-1]) < 0.2:
            continue
        new_points.append(point)
    return new_points


def to_svg_path(points):
    assert len(points) > 1
    s = f"M{points[0][0]} {-points[0][1]}"
    for point in points[1:]:
        s += f" L{point[0]} {-point[1]}"
    s += " Z"
    return f'<path d="{s}" stroke="black" fill="#eee" fill-opacity="1.0"/>'


def filter_boundaries(feature):
    def is_in_bounds(coord):
        lon, lat = coord
        # 5, 30, 65, 100
        if lat < -10 or lat > 80:
            return False
        if lon < 50 or lon > 120:
            return False
        return True
    return list(filter(
        lambda boundary: len(boundary) > 5 and any(is_in_bounds(coord) for coord in boundary),
        (
            smooth(boundary) if feature["geometry"]["type"] == "Polygon" else smooth(boundary[0])
            for boundary in feature["geometry"]["coordinates"])
    ))


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

    loss = jnp.mean(target)
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


cmap = matplotlib.colormaps["Blues"]
def svg_heat_map(values):
    start_x = 65
    start_y = 0
    width = 50
    height = 35
    values = build_static_data_selector(inputs.coords, start_y, start_y+height, start_x, start_x+width)(values) * 1000
    
    def _to_hex_color(value):
        vmin = 0
        vmax = 63
        r,g,b,a = cmap((value - vmin) / (vmax - vmin))
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    s = ""
    for dx in range(width):
        for dy in range(height):
            value = values[dy][dx]
            s += f'<rect width="1" height="1" x="{start_x+dx-0.5}" y="{-(start_y+dy)-0.5}" fill-opacity="{min(0.9, 0.9*(value/5))}" fill="{_to_hex_color(value)}" stroke="none"/>'
    return s


def svg_result(rains, paths):
    NEWLINE = "\n"
    return f'''<svg viewBox="82 -31 12 15" xmlns="http://www.w3.org/2000/svg" style="stroke-width: 0.2%" clip-path="url(#clip)">
    <defs>
        <clipPath id="clip" clipPathUnits="userSpaceOnUse">
            <rect x="0" y="0" width="100%" height="100%" />
        </clipPath>
    </defs>
    {NEWLINE.join(paths)}
    {svg_heat_map(rains)}
    <rect width="5" height="6" stroke="#555" fill="none" y="-26" x="86" stroke-dasharray="1%"/>
    <rect x="82" y="-31" width="12" height="15" stroke="black" fill="none"/>
</svg>'''

if __name__ == "__main__":
    # Country borders
    with open("data/geoBoundariesCGAZ_ADM0.geojson", "r") as f:
        boundary_data = json.load(f)
    
    all_paths = sum(
        (
            list(
                map(to_svg_path, filter_boundaries(boundaries)
                )
            )
            for boundaries in boundary_data["features"]
        ), start=[]
    )

    era5 = xarray.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
    era5 = era5.isel(time=slice(1, None, 2))
    single_sample = era5.isel(time=0).compute()
    single_sample = single_sample.rename_dims(dict(latitude="lat", longitude="lon"))
    single_sample = single_sample.rename_vars(dict(latitude="lat", longitude="lon"))
    single_sample = single_sample.isel(lat=slice(None, None, -1))
    inputs, targets, forcings = load_data(era5, "2020-05-18T06:00:00", task_config, lead_time=timedelta(days=2.5))

    variable_selector = lambda x: x["total_precipitation_12hr"]
    region_selection_fn = build_static_data_selector(inputs.coords, 20, 25, 87, 92)

    grads_fn = functools.partial(
        adv_grads_fn_jitted,
        region_selection_fn=region_selection_fn,
        variable_selection_fn=variable_selector
    )

    perturbation = our_attack(
        inputs,
        targets,
        forcings,
        0.07,
        grads_fn,
        maxiter=50,
        do_log=False,
    )
    denoised_predictions = multi_step_forward_jit(
        jax.random.PRNGKey(1234567890),
        add_perturbation(inputs, perturbation),
        targets,
        forcings,
        forward_fn_jitted,
    )

    unperturbed_rain = targets.isel(batch=0, time=-1)["total_precipitation_12hr"].data
    perturbed_rain = xarray_jax.unwrap_data(denoised_predictions.isel(batch=0, time=-1)["total_precipitation_12hr"])

    with open("data/results/rain_case_study_before.svg", "w") as f:
        f.write(svg_result(unperturbed_rain, all_paths))
    with open("data/results/rain_case_study_after.svg", "w") as f:
        f.write(svg_result(perturbed_rain, all_paths))