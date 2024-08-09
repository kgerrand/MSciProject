site = "MHD"
compound = "ch4"
model_type = "rf"

compound_list = ['ch4',
                 'cf4',
                 'cfc-12',
                 'ch2cl2',
                 'ch3br',
                 'hcfc-22',
                 'hfc-125',
                 'hfc-134a',
                 'n2o',
                 'sf6']

site_dict = {# NAME Baselines - AGAGE Network
             "MHD":"Mace Head, Ireland", 
             "RPB":"Ragged Point, Barbados", 
             "CGO":"Cape Grim, Australia", 
             "GSN":"Gosan, South Korea",
             "JFJ":"Jungfraujoch, Switzerland", 
             "CMN":"Monte Cimone, Italy", 
             "THD":"Trinidad Head, USA", 
             "ZEP":"Zeppelin, Svalbard",
             "SMO": "Cape Matatula, American Samoa"
             }


site_coords_dict = {# NAME Baselines - AGAGE Network
                    "MHD":[53.3267, -9.9046], 
                    "RPB":[13.1651, -59.4321], 
                    "CGO":[-40.6833, 144.6894], 
                    "GSN":[33.2924, 126.1616],
                    "JFJ":[46.547767, 7.985883], 
                    "CMN":[44.1932, 10.7014], 
                    "THD":[41.0541, -124.151], 
                    "ZEP":[78.9072, 11.8867],
                    "SMO": [-14.2474, -170.5644]
                    }


confidence_threshold = 0.8