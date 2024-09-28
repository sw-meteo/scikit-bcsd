import xarray as xr

def to_circular_longitude(ds, lon_name='lon'):
    ds_l = ds.isel(lon=0)
    ds_l[lon_name] = ds_l[lon_name] + 360
    ds_r = ds.isel(lon=-1)
    ds_r[lon_name] = ds_r[lon_name] - 360
    return xr.concat([ds_r, ds, ds_l], dim=lon_name)