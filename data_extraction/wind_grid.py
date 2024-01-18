'''

This script transforms the meteorological data taken from ECMWF into a grid of 3x3 points surrounding an AGAGE station. The data is then saved as a netCDF file, split into one u and v component file per year.
The code is set-up for Mace Head, Ireland. This can be easily changed by changing the site_lat and site_lon variables, as well as the data_path variable.

'''

import xarray as xr
import numpy as np
from pathlib import Path

data_path = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'/'data_files'/'meteorological_data'/'ECMWF'/'MHD'

# latitude and longitude of Mace Head, Ireland
site_lat = 53.3267
site_lon = -9.9046

# creating the grid of 3x3 points surrounding site of interest
points_lat = np.array([0, 5, 5, 0, -5, -5, -5, 0, 5]) + site_lat
points_lon = np.array([0, 0, 5, 5, 5, 0, -5, -5, -5]) + site_lon
points = range(9)

# creating an xarray DataArray for the grid coordinates
target_lat = xr.DataArray(points_lat, dims=["points"], coords={"points": points})
target_lon = xr.DataArray(points_lon, dims=["points"], coords={"points": points})


years = range(2015, 2016)

# loop through the months and years to transform the data
for fi, yr in enumerate(years):

    print(f"Opening {yr}")

    # lists to store data for each year
    u_list = []
    v_list = []

    for month in range(1, 13):
   
        # open the data for the month and year
        extracted_data = xr.open_mfdataset((data_path/'10m_wind').glob(f"*{yr}_{month:02d}.nc"))    

        # extract the u and v components of the wind
        extracted_u = extracted_data['u10']
        extracted_v = extracted_data['v10']

        # interpolate the data onto the grid
        u = extracted_u.interp(latitude=target_lat, longitude=target_lon, method="nearest")
        v = extracted_v.interp(latitude=target_lat, longitude=target_lon, method="nearest")

        # appending monthly data to year lists
        u_list.append(u)
        v_list.append(v)

    # concatenating the monthly data along the 'time' dimension
    u_combined = xr.concat(u_list, dim='time')
    v_combined = xr.concat(v_list, dim='time')

    # saving the combined data for the year
    u_combined.to_netcdf(data_path/'10m_wind_grid'/f"10m_u_{yr}.nc")
    v_combined.to_netcdf(data_path/'10m_wind_grid'/f"10m_v_{yr}.nc")

    print(f"Closing {yr}")