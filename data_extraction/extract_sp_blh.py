'''

This script transforms the surface pressure and boundary layer height data taken from ECMWF into a single file per year.
The data is saved as a netCDF file.

'''

import xarray as xr
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


years = range(1999, 2024)

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

        # selecting the data for the site of interest
        sp = sp.interp(latitude=site_lat, longitude=site_lon, method='nearest')
        blh = blh.interp(latitude=site_lat, longitude=site_lon, method='nearest')

        # appending monthly data to year lists
        sp_list.append(sp)
        blh_list.append(blh)

        #=======================================================================
    
    # concatenating the monthly data along the 'time' dimension and saving the combined data for the year
    #=======================================================================
    # surface pressure
    sp_combined = xr.concat(sp_list, dim='time')
    sp_combined.to_netcdf(data_path/'surface_pressure'/f"{site}_sp_{yr}.nc")

    # boundary layer height
    blh_combined = xr.concat(blh_list, dim='time')
    blh_combined.to_netcdf(data_path/'boundary_layer_height'/f"{site}_blh_{yr}.nc")

    #=======================================================================

    print(f"Closing {yr}")