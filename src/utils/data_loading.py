import dataclasses

from datetime import datetime, timedelta

import xarray

from google.cloud import storage

from graphcast import checkpoint
from graphcast import data_utils
from graphcast import gencast
from graphcast import nan_cleaning
from graphcast import normalization

def init_gcs():
    # anonymous gcs client for the bucket
    # bucket contains model, example data and normalization data
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    return gcs_bucket


def load_normalization_data(gcs_bucket):
    dir_prefix = "gencast/"

    # Load normalization data
    with gcs_bucket.blob(dir_prefix+"stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(dir_prefix+"stats/min_by_level.nc").open("rb") as f:
        min_by_level = xarray.load_dataset(f).compute()
    
    return diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level


def load_model(gcs_bucket):
    diffs_stddev_by_level, mean_by_level, stddev_by_level, min_by_level = load_normalization_data(gcs_bucket)

    with gcs_bucket.blob("gencast/params/GenCast 1p0deg Mini <2019.npz").open("rb") as f:
        ckpt = checkpoint.load(f, gencast.CheckPoint)
    params = ckpt.params
    state = {}

    task_config = ckpt.task_config
    sampler_config = ckpt.sampler_config
    noise_config = ckpt.noise_config
    noise_encoder_config = ckpt.noise_encoder_config
    denoiser_architecture_config = ckpt.denoiser_architecture_config

    # change to triblockdiag_mha for gpu
    denoiser_architecture_config.sparse_transformer_config.attention_type = "triblockdiag_mha"
    denoiser_architecture_config.sparse_transformer_config.mask_type = "full"

    # fix to more appropriate config
    noise_config = gencast.NoiseConfig(
        training_max_noise_level=sampler_config.max_noise_level/2,
        training_min_noise_level=sampler_config.min_noise_level,
        training_noise_level_rho=sampler_config.rho
    )

    def construct_wrapped_gencast():
        """Constructs and wraps the GenCast Predictor."""
        predictor = gencast.GenCast(
            sampler_config=sampler_config,
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )

        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )

        predictor = nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=min_by_level,
            var_to_clean='sea_surface_temperature',
        )

        return predictor
    
    return construct_wrapped_gencast, params, state, task_config


def load_data(era5, date, task_config, downsample=True, lead_time=None):
    if lead_time is None:
        lead_time = timedelta(days=2) # default for evaluation
    date_format = "%Y-%m-%dT%H:%M:%S"
    start = datetime.fromisoformat(date)
    end = start + lead_time
    batch = era5.sel(time=slice(start.strftime(date_format), end.strftime(date_format))).compute()

    # renaming
    batch.coords["datetime"] = batch.coords["time"]
    batch = batch.rename_dims(dict(latitude="lat", longitude="lon"))
    batch = batch.rename_vars(dict(latitude="lat", longitude="lon"))

    # we need to rescale the data such that it is only T06:00:00 and T18:00:00
    # era5 also includes times at 00:00 and 12:00, see gencast paper/era5 data
    # but this is now already done during caching of era5, so it is commented out
    # batch = batch \
    #     .isel(time=slice(0, None, 2))

    # also need to downsample lat/lon grid so that it is every 1deg instead of 0.25deg
    # for computational reasons
    if downsample:
        batch = batch \
            .isel(lat=slice(0, None, 4)) \
            .isel(lon=slice(0, None, 4))
    # add batch dimension
    batch = batch.expand_dims(dim=dict(batch=1), axis=0)
    batch.coords["datetime"] = batch.coords["time"].expand_dims(dim=dict(batch=1), axis=0)
    # mirror latitude
    batch = batch.isel(lat=slice(None, None, -1))

    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        batch,
        target_lead_times=slice("12h", f"{(batch.sizes['time']-2)*12}h"), # All but 2 input frames.
        **dataclasses.asdict(task_config)
    )

    return inputs, targets, forcings