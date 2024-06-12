'''
This script downloads meteorological data from the ECMWF (European Centre for Medium-Range Weather Forecasts) using the CDS (Climate Data Store) API. 
The data being downloaded has been limited to the area surrounding Mace Head in Ireland, an AGAGE data collection station (20W-0W, 40N-60N).
The data being downloaded is u and v wind at pressure levels of 500 and 850 hPa.

This code is ran on BC4, a supercomputer at the University of Bristol.

The code to download data from other sites is very similar; it just sees a change in the file path and the area being downloaded.

'''

import cdsapi
import sys
from pathlib import Path
import os

c = cdsapi.Client()


# change to extract data for different AGAGE sites
site = 'BA'

site_coords_dict = {"MHD":[53.3267, -9.9046], 
                    "BA":[13.1651, -59.4321], 
                    "CG":[-40.6833, 144.6894], 
                    "GSN":[33.2924, 126.1616],
                    "J1":[46.547767, 7.985883], 
                    "M5":[44.1932, 10.7014], 
                    "TH":[41.0541, -124.151], 
                    "ZE":[78.9072, 11.8867],
                    "SMO": [-14.2474, -170.5644]}

# extract site coordinates
site_lat = site_coords_dict[site][0]
site_lon = site_coords_dict[site][1]


# output_path = Path.home()/'Year4'/'MSciProject'/'data'/'single_levels'
output_path = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'/'data_files'/'meteorological_data'/'ECMWF'/site/'pressure_levels'

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']



def main(year):

    for n in range(len(months)):
        output_filename = f'{output_path}/{site}_3dwind_{year}_{months[n]}.nc'

        if os.path.exists(output_filename):
            print(f"{months[n]} {year} already downloaded. Skipping.")

        else:
            print(f'Downloading {months[n]} {year}')
           
            c.retrieve(
                'reanalysis-era5-pressure-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': 
                    [
                        'u_component_of_wind', 'v_component_of_wind',
                    ],
                    
                    'pressure_level': 
                    [
                        '500', '850',
                    ],
                    
                    'year': f'{year}',
                    'month': f'{months[n]}',
                    'day': 
                    [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],

                    'time': 
                    [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                        
                    'format': 'netcdf',
                    'area': [(site_lat+12),(site_lon-12), (site_lat-12),(site_lon+12),],
                },
                # change the output file name as needed
                output_filename)
                
            print(f'{months[n]} {year} downloaded')
       

if __name__ == '__main__':
    if len(sys.argv) == 2:
        YEAR = sys.argv[1]  # Use sys.argv[1] to get the first command-line argument
        main(YEAR)
    else:
        print("Usage: python {} <YEAR>".format(sys.argv[0]))