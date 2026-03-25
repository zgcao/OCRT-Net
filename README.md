# OCRT-Net: Ocean Color Radiative Transfer embeed Neural Network for water quality retrielvas

[![Paper](https://img.shields.io/badge/Paper-Inform_Geo_Under_Review-blue.svg)]([link_to_your_paper](https://www.sciencedirect.com/journal/information-geography))
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

OCRT-Net is a physics-guided deep learning framework designed for the robust retrieval of inland and coastal water quality parameters (Chl-a, SPM, CDOM). By embedding the Gordon radiative transfer model directly into the neural network architecture, OCRT-Net achieves exceptional generalization across global diverse waters.

## Multi-Sensor Support
This repository provides production-ready weights trained on the global in-situ GLORIA dataset, dynamically supporting the following satellite sensors:
- **Sentinel-3 OLCI** (12 bands)
- **Suomi-NPP / NOAA-20/21 VIIRS** (6 bands)
- **Sentinel-2 MSI** (7 bands)
## Required environments
```python
Python 3.8+; Tensorflow 2.x; numpy; pandas;
```

## Quick Start

You can retrieve high-precision water quality parameters with just 3 lines of code! 

```python
from predict import OCRT_Predictor
import pandas as pd

# 1. 初始化引擎
model = OCRT_Predictor(sensor='OLCI', weights_path='release/OCRT_Net_Production_OLCI.weights.h5')

bands = ['Rrs_413', 'Rrs_443', 'Rrs_490', 'Rrs_510', 'Rrs_560', 'Rrs_620', 
         'Rrs_665', 'Rrs_674', 'Rrs_681', 'Rrs_709', 'Rrs_754', 'Rrs_779']

# 2. 读取数据
df = pd.read_excel('your_data.xlsx'')
rrs_matrix = df[bands].values

results_dict = model.predict(rrs_matrix)

# 4. 把自动反 Log10 后的浓度拼接到原 DataFrame
df['Chla_pred']  = results_dict['Chla_predicted']
df['SPM_pred']   = results_dict['SPM_predicted']
df['ag443_pred'] = results_dict['ag440_predicted'] 
# 6. 保存导出
df.to_excel('OLCI_matchups_pred.xlsx', index=False)
df.head()
