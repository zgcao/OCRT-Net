# OCRT-Net: Ocean Color Radiative Transfer embeed Neural Network for water quality retrielvas

[![Paper](https://img.shields.io/badge/Paper-Inform_Geo_Under_Review-blue.svg)]([link_to_your_paper](https://www.sciencedirect.com/journal/information-geography))
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

OCRT-Net is a physics-guided deep learning framework designed for the robust retrieval of inland and coastal water quality parameters (Chl-a, SPM, CDOM). By embedding the Gordon radiative transfer model directly into the neural network architecture, OCRT-Net achieves exceptional generalization across global diverse waters.

## Multi-Sensor Support
This repository provides production-ready weights trained on the global in-situ GLORIA dataset, dynamically supporting the following satellite sensors:
- **Sentinel-3 OLCI** (12 bands)
- **Suomi-NPP / NOAA-20 VIIRS** (6 bands)
- **Sentinel-2 MSI** (7 bands)

## Quick Start

You can retrieve high-precision water quality parameters with just 3 lines of code! 

```python
from predict import OCRT_Predictor

# 1. Initialize the predictor for your target sensor ('OLCI', 'VIIRS', or 'MSI')
# The script will automatically load the corresponding weights and physical bases.
model = OCRT_Predictor(sensor='VIIRS')

# 2. Run inference directly on your CSV file
results = model.predict_from_csv(input_csv='your_viirs_data.csv', output_csv='output_results.csv')
