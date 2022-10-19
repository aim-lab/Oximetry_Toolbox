# PhysioZoo OBM documentation

Oximetry digital biomarkers for the analysis of continuous oximetry (SpO2) time series.

Based on the paper 
Jeremy Levy, Daniel ́Alvarez, Aviv A Rosenberg, Alexandra Alexandrovich, F ́elix Del Campo, and Joachim ABehar.  Digital oximetry biomarkers for assessing respiratory function:  standards of measurement, physiologicalinterpretation, and clinical use.NPJ digital medicine, 4(1):1–14, 2021

..  youtube:: 1m0nQ4MIOdE
    :width: 640
    :height: 480
    
## Description

Five types of biomarkers may be evaluated:

1.  General statistics: time-based statistics describing the oxygen saturation time series data distribution.

2.  Complexity: quantify the presence of long-range correlations in non-stationary time series.

3.  Periodicity: quantify consecutive events creating some periodicity in the oxygen saturation time series.

4.  Desaturations: time-based measures that are descriptive statistics of the desaturation patterns happening throughout the time series.

5.  Hypoxic burden: time-based measures quantifying the overall degree of hypoxemia imposed to the heart and other organs during the recording period.

## Installation

Available on pip, with the command: 
pip install pobm

pip project: https://pypi.org/project/pobm/

## Requirements

numpy > 1.18.2

scikit-learn > 0.22.2

scipy > 1.4.1

lempel-ziv-complexity==0.2.2

All the requirements are installed when the toolbox is installed, no need for additional commands.

## Documentation

Available at https://oximetry-toolbox.readthedocs.io/en/latest/
