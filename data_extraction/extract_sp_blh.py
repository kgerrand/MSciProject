'''

This script transforms the surface pressure and boundary layer height data taken from ECMWF into a single file per year.
The data is saved as a netCDF file.

'''

import xarray as xr
import numpy as np
from pathlib import Path

import sys
sys.path.append('../')
import config

site = 'MHD'
site_name = config.site_dict[site]
data_path = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'/'data_files'/'meteorological_data'/'ECMWF'/site


print(f"Extracting data collected in {site_name}.")


years = range(1978, 1982)

# loop through the months and years to transform the data
for fi, yr in enumerate(years):

    print(f"Opening {yr}")

    # lists to store data for each year for surface pressure and boundary layer height
    sp_list = []
    blh_list = []


    for month in range(1, 13):
        #=======================================================================
        # extracting the data for each month and year
        extracted_data = xr.open_mfdataset((data_path/'single_levels').glob(f"*{yr}_{month:02d}.nc"))    

        # extracting variables
        sp = extracted_data['sp']
        blh = extracted_data['blh']

        # appending monthly data to year lists
        sp_list.append(sp)
        blh_list.append(blh)

        #=======================================================================
    
    # concatenating the monthly data along the 'time' dimension and saving the combined data for the year
    #=======================================================================
    # surface pressure
    sp_combined = xr.concat(sp_list, dim='time')
    sp_combined.to_netcdf(data_path/'surface_pressure'/f"sp_{yr}.nc")

    # boundary layer height
    blh_combined = xr.concat(blh_list, dim='time')
    blh_combined.to_netcdf(data_path/'boundary_layer_height'/f"blh_{yr}.nc")

    #=======================================================================

    print(f"Closing {yr}")