'''

This script transforms the meteorological data taken from ECMWF into a grid of 2 sets of 3x3 (+/- 5 and 10 lat/lon) points surrounding an AGAGE station, at three different heights.
The data is saved as a netCDF file, split into one u and v component file per year.

'''

import xarray as xr
import numpy as np
from pathlib import Path

import sys
sys.path.append('../')

site = 'BA'

site_coords_dict = {"MHD":[53.3267, -9.9046], 
                    "RPB":[13.1651, -59.4321], 
                    "CGO":[-40.6833, 144.6894], 
                    "GSN":[33.2924, 126.1616],
                    "JFJ":[46.547767, 7.985883], 
                    "CMN":[44.1932, 10.7014], 
                    "THD":[41.0541, -124.151], 
                    "ZEP":[78.9072, 11.8867],
                    "SMO": [-14.2474, -170.5644]}

site_lat = site_coords_dict[site][0]
site_lon = site_coords_dict[site][1]


print(f"Creating grid for data collected in {site}.")


data_path = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'/'data_files'/'meteorological_data'/'ECMWF'/site


# creating a grid system with +/- 5 latitude and longitude from the site of interest
# points_lat = np.array([0, 5, 5, 0, -5, -5, -5, 0, 5]) + site_lat
# points_lon = np.array([0, 0, 5, 5, 5, 0, -5, -5, -5]) + site_lon
# points = range(9)

# creating a grid system with +/- 5 and 10 degrees latitude and longitude from the site of interest
points_lat = np.array([0, 5, 5, 0, -5, -5, -5, 0, 5, 10, 10, 0, -10, -10, -10, 0, 10]) + site_lat
points_lon = np.array([0, 0, 5, 5, 5, 0, -5, -5, -5, 0, 10, 10, 10, 0, -10, -10, -10]) + site_lon
points = range(17)


# creating an xarray DataArray for the grid coordinates
target_lat = xr.DataArray(points_lat, dims=["points"], coords={"points": points})
target_lon = xr.DataArray(points_lon, dims=["points"], coords={"points": points})


years = range(1999, 2024)

# loop through the months and years to transform the data
for fi, yr in enumerate(years):

    print(f"Opening {yr}")

    # lists to store data for each year for each of the three levels
    u_list_10m = []
    v_list_10m = []

    u_list_850hpa = []
    v_list_850hpa = []

    u_list_500hpa = []
    v_list_500hpa = []

    for month in range(1, 13):
        # extracting the data for each month and year
        #=======================================================================
        # 10m wind
        # open the data for the month and year
        extracted_data = xr.open_mfdataset((data_path/'single_levels').glob(f"*{yr}_{month:02d}.nc"))

        # extract the u and v components of the wind
        extracted_u = extracted_data['u10']
        extracted_v = extracted_data['v10']

        # interpolate the data onto the grid
        u = extracted_u.interp(latitude=target_lat, longitude=target_lon, method="nearest")
        v = extracted_v.interp(latitude=target_lat, longitude=target_lon, method="nearest")

        # appending monthly data to year lists
        u_list_10m.append(u)
        v_list_10m.append(v)

        #=======================================================================
        # 850hPa wind
        extracted_data = xr.open_mfdataset((data_path/'pressure_levels').glob(f"*{yr}_{month:02d}.nc")).sel(level=850)

        extracted_u = extracted_data['u']
        extracted_v = extracted_data['v']

        u = extracted_u.interp(latitude=target_lat, longitude=target_lon, method="nearest")
        v = extracted_v.interp(latitude=target_lat, longitude=target_lon, method="nearest")

        u_list_850hpa.append(u)
        v_list_850hpa.append(v)

        #=======================================================================
        # 500hPa wind
        extracted_data = xr.open_mfdataset((data_path/'pressure_levels').glob(f"*{yr}_{month:02d}.nc")).sel(level=500)

        extracted_u = extracted_data['u']
        extracted_v = extracted_data['v']

        u = extracted_u.interp(latitude=target_lat, longitude=target_lon, method="nearest")
        v = extracted_v.interp(latitude=target_lat, longitude=target_lon, method="nearest")

        u_list_500hpa.append(u)
        v_list_500hpa.append(v)
        #=======================================================================
        
    
    # concatenating the monthly data along the 'time' dimension and saving the combined data for the year
    #=======================================================================
    # 10m wind
    u_combined_10m = xr.concat(u_list_10m, dim='time')
    v_combined_10m = xr.concat(v_list_10m, dim='time')

    u_combined_10m.to_netcdf(data_path/'10m_wind_grid'/f"{site}_10m_u_{yr}.nc")
    v_combined_10m.to_netcdf(data_path/'10m_wind_grid'/f"{site}_10m_v_{yr}.nc")

    #=======================================================================
    # 850hPa wind
    u_combined_850hpa = xr.concat(u_list_850hpa, dim='time')
    v_combined_850hpa = xr.concat(v_list_850hpa, dim='time')

    u_combined_850hpa = xr.DataArray(u_combined_850hpa, name='u850')
    v_combined_850hpa = xr.DataArray(v_combined_850hpa, name='v850')

    u_combined_850hpa.to_netcdf(data_path/'850hPa_wind_grid'/f"{site}_850hPa_u_{yr}.nc")
    v_combined_850hpa.to_netcdf(data_path/'850hPa_wind_grid'/f"{site}_850hPa_v_{yr}.nc")

    #=======================================================================
    # 500hPa wind
    u_combined_500hpa = xr.concat(u_list_500hpa, dim='time')
    v_combined_500hpa = xr.concat(v_list_500hpa, dim='time')

    u_combined_500hpa = xr.DataArray(u_combined_500hpa, name='u500')
    v_combined_500hpa = xr.DataArray(v_combined_500hpa, name='v500')

    u_combined_500hpa.to_netcdf(data_path/'500hPa_wind_grid'/f"{site}_500hPa_u_{yr}.nc")
    v_combined_500hpa.to_netcdf(data_path/'500hPa_wind_grid'/f"{site}_500hPa_v_{yr}.nc")

    #=======================================================================

    print(f"Closing {yr}")