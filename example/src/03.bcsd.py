'''
降水降尺度历史试验
变量: pet_ann (全年潜在蒸散发)
空间: 全球，只对陆地上的格点计算
方法: BCSD
  9-yr running detrend
  BC additive & SD multiplicative
数据: 历史模拟 + GPCC观测
输入模式名称和训练、验证时段
e.g. `python ~.py ACCESS-ESM1-5 1921 2014 1901 1920`

BCSD downscaling of historical precipitation
only cauculate on land grid points
detrend when doing quantile mapping

'''

import xarray as xr
import pandas as pd
import numpy as np
import rootutils as rt
import matplotlib.pyplot as plt
import matplotlib.colors
import os
from tqdm import tqdm
import sys
import warnings
warnings.filterwarnings("ignore")

from utils import remove_nan, to_circular_longitude
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models import BcsdPrecipitation
from skdownscale.spatial_models import SpatialDisaggregator


'''
0. settings
'''
model_name = "ACCESS-ESM1-5"
obs_name = "GPCC"
var_name = "pre"
train_range = slice("1921", "2014")
pred_range = slice("1901", "1920")
plot_result = True
save_with_mask = True

base = rt.find_root(search_from=__file__, indicator=".project-root")
ifdir = "interim"
ofdir = "result"
os.makedirs(os.path.join(base, ofdir), exist_ok=True)

ifname_model = f"{var_name}.{model_name}.1901-2014.nc"
ifname_obs = f"{var_name}.{obs_name}.1901-2014.nc"
ofname_bcsd = f"{var_name}.{model_name}.{pred_range.start}-{pred_range.stop}.bcsd.nc"
ofname_target = f"{var_name}.{obs_name}.{pred_range.start}-{pred_range.stop}.bcsd.nc"
ifname_model_VG = f"{var_name}.valid_grid.{obs_name}.on.{model_name}.nc"
ifname_obs_VG = f"{var_name}.valid_grid.{obs_name}.nc"

print(f"processing {var_name} of {model_name}\n \
    train range: {train_range.start}-{train_range.stop}\n \
    pred range: {pred_range.start}-{pred_range.stop}")


'''
1. load data
'''
print("load data ...", end=" ")

ds_obs = xr.open_dataset(os.path.join(base, ifdir, ifname_obs))
ds_model = xr.open_dataset(os.path.join(base, ifdir, ifname_model))

ds_obs_train = ds_obs.sel(time=train_range)
ds_obs_target = ds_obs.sel(time=pred_range)

ds_model_train = ds_model.sel(time=train_range)
ds_model_pred = ds_model.sel(time=pred_range)
fine_valid_grid = xr.open_dataset(os.path.join(base, ifdir, ifname_obs_VG))
coarse_valid_grid = xr.open_dataset(os.path.join(base, ifdir, ifname_model_VG))

print("done")


'''
2. bias correction
'''
print("bias correction ...", end=" ")

ds_obs_on_model_grid = to_circular_longitude(ds_obs_train).interp_like(ds_model_train, method='linear')
ds_target_on_model_grid = to_circular_longitude(ds_obs_target).interp_like(ds_model_pred, method='linear')

res_pred = np.empty((ds_model_pred.time.size,
    ds_model_train.lat.size, ds_model_train.lon.size))
res_train = np.empty((ds_model_train.time.size,
    ds_model_train.lat.size, ds_model_train.lon.size))

for i_lon, lon in enumerate(tqdm(ds_model_train.lon, position=0, leave=True)):
    for i_lat, lat in enumerate(ds_model_train.lat):
        if coarse_valid_grid.valid_grid[i_lat, i_lon]:
            y_train = ds_obs_on_model_grid.isel(lat=i_lat, lon=i_lon)[var_name]
            X_train = ds_model_train.isel(lat=i_lat, lon=i_lon)[var_name]
            X_pred = ds_model_pred.isel(lat=i_lat, lon=i_lon)[var_name]
            bcsd_model = PointWiseDownscaler(BcsdPrecipitation(qm_kwargs={"detrend": True}, return_anoms=False), dim='time')
            bcsd_model.fit(X_train, y_train)
            out_train = bcsd_model.predict(X_train)
            out_pred = bcsd_model.predict(X_pred)
            
            res_pred[:, i_lat, i_lon] = out_pred.values
            res_train[:, i_lat, i_lon] = out_train.values

for i_time in range(res_pred.shape[0]):
    res_pred[i_time, ~coarse_valid_grid.valid_grid.values] = np.nan
for i_time in range(res_train.shape[0]):
    res_train[i_time, ~coarse_valid_grid.valid_grid.values] = np.nan
    
ds_model_train_bc = xr.Dataset({var_name: 
    xr.DataArray(res_train, 
                dims=('time', 'lat', 'lon'), 
                coords={'time': ds_model_train.time, 'lat': ds_model_train.lat, 'lon': ds_model_train.lon})})
ds_model_pred_bc = xr.Dataset({var_name: 
    xr.DataArray(res_pred, 
                dims=('time', 'lat', 'lon'), 
                coords={'time': ds_model_pred.time, 'lat': ds_model_pred.lat, 'lon': ds_model_pred.lon})})

print("done")


'''
3. spatial disaggregation
'''
print("spatial disaggregation ...", end=" ")

ds_obs_train = remove_nan(ds_obs_train, var_name)
climo_fine = ds_obs_train.groupby('time.month').mean('time')

ds_model_pred_bc = remove_nan(ds_model_pred_bc, var_name)
ds_model_train_bc = remove_nan(ds_model_train_bc, var_name)
climo_coarse = ds_model_train_bc.groupby('time.month').mean('time')

bcsd_spatial = SpatialDisaggregator(var=var_name, var_like="precipitation")

bcsd_spatial.fit(ds_model_pred_bc, climo_coarse, var_name=var_name,var_like="precipitation", temporal_resolution="monthly")
ds_model_pred_bcsd = bcsd_spatial.predict(climo_fine, var_name=var_name, var_like="precipitation", temporal_resolution="monthly")
ds_model_pred_bcsd = ds_model_pred_bcsd.drop_vars('month')

if save_with_mask:
    ds_model_pred_bcsd = ds_model_pred_bcsd.where((fine_valid_grid.valid_grid.values))
else:
    ds_model_pred_bcsd = ds_model_pred_bcsd.where((fine_valid_grid.valid_grid.values), -999)

print("done")


'''
4. save data
'''
encoding = {'lat': {'_FillValue': False},
            'lon': {'_FillValue': False},
            var_name: {'_FillValue': -999, 
                    'dtype': 'float32'}}
ds_model_pred_bcsd.astype('float').to_netcdf(os.path.join(base, ofdir, ofname_bcsd), encoding=encoding)
ds_obs_target.astype('float').to_netcdf(os.path.join(base, ofdir, ofname_target), encoding=encoding)

print("results saved")


'''
5. plot (optional)
'''
if plot_result:
    levels = np.arange(0, 300, 10)
    cmap = "gist_earth_r"

    fig = plt.figure(figsize=(20, 3.5), dpi=300)
    grid = plt.GridSpec(1, 13, wspace=1,)
    ax0 = fig.add_subplot(grid[0, 0:3])
    ax1 = fig.add_subplot(grid[0, 3:6])
    ax2 = fig.add_subplot(grid[0, 6:9])
    ax_cb = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0, 0.01, ax2.get_position().height])


    ctf = ds_model_pred.where(coarse_valid_grid.valid_grid.values)[var_name].mean(dim='time').\
        plot.contourf(ax=ax0, cmap=cmap,levels=levels, extend='both', add_colorbar=False)
    ax0.set_title(f'{model_name} {pred_range.start}-{pred_range.stop}')
    ctf = ds_obs_target.where(fine_valid_grid.valid_grid.values)[var_name].mean(dim='time').\
        plot.contourf(ax=ax1, cmap=cmap, levels=levels, extend='both', add_colorbar=False)
    ax1.set_title('obs target')
    ctf = ds_model_pred_bcsd.where(fine_valid_grid.valid_grid.values)[var_name].mean(dim='time').\
        plot(ax=ax2, cmap=cmap, levels=levels, extend='both', add_colorbar=False)
    ax2.set_title('after bcsd')
    fig.colorbar(ctf, cax=ax_cb, orientation='vertical', label='pre (mm/year)')

    fig.savefig(os.path.join(base, ofdir, f"{var_name}.{model_name}.{pred_range.start}-{pred_range.stop}.png"), dpi=300, bbox_inches='tight')
    print("plot saved")
