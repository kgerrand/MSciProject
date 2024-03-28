This repository contains all code that was developed during my MSci resarch project in Atmospheric Computational Chemistry undertaken in my final year of my Chemistry with Scientific Computing degree (23-24).


### Context
The aim of the project was to use machine learning to accurately identify baseline data points in a greenhouse gas concentration time series. These predictions are made independent of species and based on meteorological conditions, and are based on outputs obtained from the UK Met Office. In this context, baseline refers to data points that are representative of background conditions at a given latitude. 

I did this work as part of the Atmospheric Chemistry Research Group (ACRG) at the University of Bristol under the supervision of Prof. Matt Rigby.

### Running the Code
To run the code, the required dataset must first be created using [baseline_setup.ipynb](https://github.com/kgerrand/MSciProject/blob/main/baselines_setup.ipynb). This collects the relevant meteorology, concentration data and baseline flags. The meteorological data were taken from the [EMCWF Era5 reanalyses](https://cds.climate.copernicus.eu/#!/search?text=ERA5&type=dataset&keywords=((%20%22Product%20type:%20Reanalysis%22%20)%20AND%20(%20%22Variable%20domain:%20Atmosphere%20(surface)%22%20)%20AND%20(%20%22Spatial%20coverage:%20Global%22%20)%20AND%20(%20%22Temporal%20coverage:%20Past%22%20)%20AND%20(%20%22Provider:%20Copernicus%20C3S%22%20))), and the concentration from [AGAGE](https://agage.mit.edu/).
Following the creation of the dataset, the models (as defined in [final models](https://github.com/kgerrand/MSciProject/tree/main/models)) are tested through [quantitative and qualitative evaluation](https://github.com/kgerrand/MSciProject/blob/main/model_evaluation/model_eval.ipynb), as well as by [comparison to a benchmark](https://github.com/kgerrand/MSciProject/blob/main/model_evaluation/benchmark_comparison.ipynb).
My findings are summarised in the associated thesis.