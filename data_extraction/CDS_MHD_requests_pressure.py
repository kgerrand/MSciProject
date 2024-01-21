'''
This script downloads meteorological data from the ECMWF (European Centre for Medium-Range Weather Forecasts) using the CDS (Climate Data Store) API. 
The data being downloaded has been limited to the area surrounding Mace Head in Ireland, an AGAGE data collection station (20W-0W, 40N-60N).
The data being downloaded is u and v wind at pressure levels of 500 and 850 hPa.

This code is ran on BC4, a supercomputer at the University of Bristol.

'''

import cdsapi
import sys
from pathlib import Path

c = cdsapi.Client()

output_path = Path.home()/'Year4'/'MSciProject'/'data'/'pressure_levels'

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
names = ['Jan','Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']


def main(year):

    for n in range(len(months)):
        print(f'Downloading {names[n]} {year}')
           
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
                'area': [60,-20, 40,0,],
            },
            # change the output file name as needed
            f'{output_path}/3dwind_{year}_{months[n]}.nc')
            
        print(f'{names[n]} {year} downloaded')
       

if __name__ == '__main__':
    if len(sys.argv) == 2:
        YEAR = sys.argv[1]  # Use sys.argv[1] to get the first command-line argument
        main(YEAR)
    else:
        print("Usage: python {} <YEAR>".format(sys.argv[0]))