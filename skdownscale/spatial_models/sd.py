import numpy as np
import pandas as pd
import xarray as xr
from utils import to_circular_longitude


class SpatialDisaggregator:
    """
    Spatial disaggregation model class
    Apply spatial disaggregation algorithm to an xarray Dataset with fit
    and predict methods using NASA-NEX method for spatial disaggregation
    (see Thrasher et al, 2012).

    Parameters
    ----------
    var : str
          specifies the variable being downscaled. Default is
          temperature and other option is precipitation.
    """

    def __init__(self, var="temperature", var_like=None):
        self._var = var
        
        if var not in ['temperature', 'precipitation']: 
            if (var_like is None) or (var_like not in ['temperature', 'precipitation']):
                raise NotImplementedError(
                    "functionality for spatial disaggregation"
                    " of %s has not yet been added" % var
                )
        self.sfc = None; self.fitted = False

    def fit(self, ds_bc, climo_coarse, var_name, var_like=None, temporal_resolution="daily", lat_name="lat", lon_name="lon"):
        """
        Fit the scaling factor used in spatial disaggregation

        Parameters
        -----------
        ds_bc : xarray.Dataset
                Daily bias corrected data at the model resolution
        climo_coarse : xarray.DataArray or xarray.Dataset
                Observed climatology that has been regridded (coarsened)
                to the model resolution
        var_name : str
                  Specifies the data variable name within ds_bc of the
                  daily bias corrected data
        lat_name : str
                  Name of the latitude dimension of ds_bc and climo_coarse,
                  default is 'lat'.
        lon_name : str
                  Name of the longitude dimension of ds_bc and climo_coarse,
                  default is 'lon'.
        """

        # check that climo has been regridded to model res
        if not ds_bc[lat_name].equals(climo_coarse[lat_name]):
            raise ValueError("climo latitude dimension does not match model res")
        if not ds_bc[lon_name].equals(climo_coarse[lon_name]):
            raise ValueError("climo longitude dimension does not match model res")

        scf = self._calculate_scaling_factor(ds_bc, climo_coarse, var_name, var_like, temporal_resolution)

        self.scf = scf
        self.fitted = True

    def predict(self, climo_fine, var_name, var_like=None, temporal_resolution="daily", lat_name="lat", lon_name="lon"):
        """
        Predict (apply) the scaling factor to the observed climatology.

        Parameters
        -----------
        scf : xarray.Dataset
              Scale factor that has been regridded to the resolution of
              the observed data.
        climo_fine : xarray.DataArray or xarray.Dataset
                     Observed climatology at its native resolution
        var_name : str
                   Specifies the data variable name within ds_bc of the
                   daily bias corrected data
        lat_name : str
                   Name of the latitude dimension of ds_bc and climo_coarse,
                   default is 'lat'.
        lon_name : str
                   Name of the longitude dimension of ds_bc and climo_coarse,
                   default is 'lon'
        """
        assert self.fitted, "model must be fit before predict can be called"
        scf_expand = to_circular_longitude(self.scf)
        scf = scf_expand.interp(lon=climo_fine.lon, lat=climo_fine.lat, method="linear")
        
        # check that scale factor has been regridded to obs res
        if not np.array_equal(scf[lat_name], climo_fine[lat_name]):
            raise ValueError("scale factor latitude dimension does not match obs res")
        if not np.array_equal(scf[lon_name], climo_fine[lon_name]):
            raise ValueError("scale factor longitude dimension does not match obs res")

        downscaled = self._apply_scaling_factor(scf, climo_fine, var_name, var_like, temporal_resolution)

        return downscaled

    def _calculate_scaling_factor(self, ds_bc, climo, var_name, var_like=None, temporal_resolution="daily"):
        """
        compute scaling factor
        """
        # Necessary workaround to xarray's check with zero dimensions
        # https://github.com/pydata/xarray/issues/3575
        da = ds_bc[var_name]
        if sum(da.shape) == 0:
            return da
        if temporal_resolution == "daily":
            groupby_type = ds_bc.time.dt.dayofyear
        elif temporal_resolution == "monthly":
            groupby_type = ds_bc.time.dt.month
        gb = da.groupby(groupby_type)

        if (var_name == "temperature") or (var_like == "temperature"):
            return gb - climo
        elif (var_name == "precipitation") or (var_like == "precipitation"):
            return gb / climo

    def _apply_scaling_factor(self, scf, climo, var_name, var_like=None, temporal_resolution="daily"):
        """
        apply scaling factor
        """
        if temporal_resolution == "daily":
            groupby_type = scf.time.dt.dayofyear
        elif temporal_resolution == "monthly":
            groupby_type = scf.time.dt.month
        da = scf[var_name]
        sff_daily = da.groupby(groupby_type)

        if (var_name == "temperature") or (var_like == "temperature"):
            return sff_daily + climo
        elif (var_name == "precipitation") or (var_like == "precipitation"):
            return sff_daily * climo
