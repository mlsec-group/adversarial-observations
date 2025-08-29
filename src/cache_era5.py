import os
import xarray

print("Starting opening")
era5 = xarray.open_zarr("gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
print("Starting selection")
full_year = era5.sel(time=slice("2022-01-01T06:00:00", "2023-01-03T18:00:00", 2))

all_vars = { # input_variables | forcing_variables | target_variables
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_temperature',
    'day_progress_cos',
    'day_progress_sin',
    'geopotential',
    'geopotential_at_surface',
    'land_sea_mask',
    'mean_sea_level_pressure',
    'sea_surface_temperature',
    'specific_humidity',
    'temperature',
    'total_precipitation_12hr',
    'u_component_of_wind',
    'v_component_of_wind',
    'vertical_velocity',
    'year_progress_cos',
    'year_progress_sin'
}
selected_full_year = full_year[list(all_vars & set(full_year.data_vars))]
print("Starting saving")
os.makedirs("data", exist_ok=True)
selected_full_year.to_netcdf("data/one_year_era5.nc", format="NETCDF4")