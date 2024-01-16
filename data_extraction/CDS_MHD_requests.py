'''
This script downloads meteorological data from the ECMWF (European Centre for Medium-Range Weather Forecasts) using the CDS (Climate Data Store) API. 
The data being downloaded has been limited to the area surrounding Mace Head in Ireland, an AGAGE data collection station (20W-0W, 40N-60N).

'''

import cdsapi
from pathlib import Path

c = cdsapi.Client()

output_path = Path.home()/'OneDrive'/'Kirstin'/'Uni'/'Year4'/'MSciProject'/'data_files'/'meteorological_data'/'ECMWF'/'MHD'/'10m_wind'

# change the months and years as needed
months = ['01', '02', '03', '04', '05', '06']
names = ['Jan','Feb', 'March', 'April', 'May', 'June']
years = ['2015']


# loop through the months and years to download the data
for i in range(len(years)):
    for n in range(len(months)):
        print(f'Downloading {names[n]} {years[i]}')
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': 
                [
                    # change the variables as needed
                    '10m_u_component_of_wind', '10m_v_component_of_wind',
                ],

                'year': f'{years[i]}',
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
            f'{output_path}/10m_{years[i]}_{months[n]}.nc')
        
        print(f'{names[n]} {years[i]} downloaded')
    
    print(f'All {years[i]} downloaded')