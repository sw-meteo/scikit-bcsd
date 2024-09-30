'''
Align model and obs data
- set varname: in case obs & model have different varnames
- set time axis settings: one data point per month, reset the dates to the beginning of each month. 
  cropped to the overlapping time period (1901-2014)
- set longitude format: -180 to 180”
'''

import xarray as xr
import pandas as pd
import rootutils as rt
import sys
import os

'''
0. info & read
'''
base = rt.find_root(search_from=__file__, indicator=".project-root")
ifdir = "data"
ofdir = "interim"
os.makedirs(os.path.join(base, ofdir), exist_ok=True)

model_name = "ACCESS-ESM1-5"
obs_name = "GPCC"
out_varname = "pre"
obs_varname = "pre"
model_varname = "pre"

ifname_model = f"cmip/{model_varname}-{model_name}-historical-1850-2014.nc"
ifname_obs = f"obs/{obs_varname}-{obs_name}-1901-2020.nc"
ofname_model = f"{out_varname}.{model_name}.1901-2014.nc"
ofname_obs = f"{out_varname}.{obs_name}.1901-2014.nc"
ofname_model_VG = f"{out_varname}.valid_grid.{obs_name}.on.{model_name}.nc"
ofname_obs_VG = f"{out_varname}.valid_grid.{obs_name}.nc"

ds_obs = xr.open_dataset(os.path.join(base, ifdir, ifname_obs))
ds_obs = ds_obs.assign_coords({"lon": (ds_obs.lon + 180) % 360 - 180})
ds_obs = ds_obs.sortby(ds_obs.lon)
ds_model = xr.open_dataset(os.path.join(base, ifdir, ifname_model))
ds_model = ds_model.assign_coords({"lon": (ds_model.lon + 180) % 360 - 180})
ds_model = ds_model.sortby(ds_model.lon)


'''
1. assign time coord & extract lsmask (valid_grid / VG)
'''
time_coord = pd.date_range(start='1901-01-01', periods=(2020-1901+1)*12, freq=pd.DateOffset(months=1))
ds_obs_new = xr.Dataset({out_varname: 
    xr.DataArray(ds_obs[obs_varname].values, 
                 dims=('time', 'lat', 'lon'), 
                 coords={'time': time_coord, 'lat': ds_obs.lat.values, 'lon': ds_obs.lon.values})})
ds_obs_new = ds_obs_new.sel(time=slice('1901','2014'))
valid_grid_obs = ds_obs[obs_varname].notnull().all(axis=0).rename("valid_grid")

time_coord = pd.date_range(start='1850-01-01', periods=(2014-1850+1)*12, freq=pd.DateOffset(months=1))
ds_model_new = xr.Dataset({out_varname: 
    xr.DataArray(ds_model[model_varname].values, 
                 dims=('time', 'lat', 'lon'), 
                 coords={'time': time_coord, 'lat': ds_model.lat.values, 'lon': ds_model.lon.values})})
ds_model_new = ds_model_new.sel(time=slice('1901','2014'))
valid_grid_model = ds_obs.interp(lat=ds_model.lat, lon=ds_model.lon)[obs_varname].notnull().all(axis=0).rename("valid_grid")

valid_grid_obs.to_netcdf(os.path.join(base, ofdir, ofname_obs_VG))
valid_grid_model.to_netcdf(os.path.join(base, ofdir, ofname_model_VG))
ds_obs_new.to_netcdf(os.path.join(base, ofdir, ofname_obs))
ds_model_new.to_netcdf(os.path.join(base, ofdir, ofname_model))