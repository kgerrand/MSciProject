'''
This script downloads meteorological data from the ECMWF (European Centre for Medium-Range Weather Forecasts) using the CDS (Climate Data Store) API. 
The data being downloaded has been limited to the area surrounding Mace Head in Ireland, an AGAGE data collection station (20W-0W, 40N-60N).
The data being downloaded is u and v wind at 10 m above the surface, surface pressure and boundary layer height.

This code is ran on BC4, a supercomputer at the University of Bristol.

The code to download data from other sites is very similar; it just sees a change in the file path and the area being downloaded.

'''

import cdsapi
import sys
from pathlib import Path
import os

c = cdsapi.Client()

output_path = Path.home()/'Year4'/'MSciProject'/'data'/'single_levels'

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
names = ['Jan','Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

#MHD
site_lat = 53.3267
site_lon = -9.9046


def main(year):

    for n in range(len(months)):
        output_filename = f'{output_path}/single_level_{year}_{months[n]}.nc'

        # checking if file already exists and skipping if so
        if os.path.exists(output_filename):
            print(f"{names[n]} {year} already downloaded. Skipping.")

        else:
            print(f'Downloading {names[n]} {year}')
           
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': 
                    [
                        '10m_u_component_of_wind', '10m_v_component_of_wind',
                        'surface_pressure', 'boundary_layer_height',
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
                output_filename)
                
            print(f'{names[n]} {year} downloaded')
       

if __name__ == '__main__':
    if len(sys.argv) == 2:
        YEAR = sys.argv[1]
        main(YEAR)
    else:
        print("Usage: python {} <YEAR>".format(sys.argv[0]))