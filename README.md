# OBM Toolbox

Oximetry digital biomarkers for the analysis of continuous oximetry (SpO2) recordings.

Based on the paper Levy Jeremy, Álvarez Daniel, Rosenberg Aviv A., del Campo Felix and Behar Joachim A. "Oximetry digital biomarkers for assessing respiratory function during sleep: standards of measurement, physiological interpretation, and clinical use". 
Under review in Nature Digital Medicine.

## Description

5 types of biomarkers are extracted:

General Statistics: time-based statistics describing the oxygen saturation time series data distribution.

Complexity: quantify the presence of long-range correlations in non-stationary time series.

Periodicity: quantify consecutive events creating some periodicity in the oxygen saturation time series.

Desaturations: time-based measures that are descriptive statistics of the desaturation patterns happening throughout the time series.

Hypoxic burden: time-based measures quantifying the overall degree of hypoxemia imposed to the heart and other organs during the recording period.

## Installation

Available on pip, with the command: 
python -m pip install --extra-index-url https://test.pypi.org/simple/ pobm

## Requirements

numpy==1.18.2

scikit-learn==0.22.2

scipy==1.4.1

lempel-ziv-complexity==0.2.2

All the requirements are installed when the toolbox is installed, no need for additional commands.

## Documentation

Available at https://oximetry-toolbox.readthedocs.io/en/latest/
