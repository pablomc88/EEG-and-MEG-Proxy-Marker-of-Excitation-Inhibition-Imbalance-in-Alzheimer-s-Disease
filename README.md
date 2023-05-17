# **Summary**

This repository contains the code used to compute 1/f slopes and E/I predictions from EEG/MEG data recordings that supports the findings of the following study:

**Martínez-Cañada, P., Perez-Valero, E., Minguillon, J., Pelayo, F., López-Gordo, M. A., & Morillas, C. (2022). Combining Aperiodic 1/f Slopes and Brain Simulation: An EEG/MEG Proxy Marker of Excitation/Inhibition Imbalance in Alzheimer’s Disease. bioRxiv, 2022-12. https://www.biorxiv.org/content/10.1101/2022.12.21.521529v2**

# **Usage**

Scripts to process neuroimaging data and generate slope and E/I prediction results of the research article above can be found as Jupyter notebooks in folders **EEG** and **MEG** with the file name **slopes_EI_computation.ipynb**. The folder **regression_coefficients** contains coefficients of the polynomial regression model used to infer the E/I ratio. Simulation data from the cortical network model used to fit the regression model are located in the folder **spectral_features**. 

Raw EEG data recordings have been included in **EEG/data/PLOSONE2020_DATA_v1.1.csv**. Raw MEG data recordings are available on request from the Dementia Platform UK (DPUK) (https://portal.dementiasplatform.uk/Apply). However, we include here the 1/f slopes and power spectra computed from the raw MEG data, in the folders **MEG/slopes/BioFind_data** and **MEG/spectra/BioFind_data**.

# **Dependencies** 

The scripts included in this repository have the following required Python package's dependencies (in brackets, the package's version used):
- numpy (1.22.4)
- scipy (1.8.1)
- sklearn (1.1.1)
- matplotlib (3.5.2)
- pymp (0.2)
- pandas (1.4.2)
- csv (1.0)
