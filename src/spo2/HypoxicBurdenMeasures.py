import numpy as np

from ErrorHandler import check_shape
from HypoxicBurdenMeasures_ import CompHBMeasures_, CompCA_, CompCT_
from ResultsClasses import HypoxicBurdenMeasuresResults


def HypoxicBurdenMeasures(signal, begin, end, CT_Threshold=90, CA_Baseline=None) -> HypoxicBurdenMeasuresResults:
    """
    Function that calculates Hypoxic Burden Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        begin: List of indices of beginning of each desaturation event.
        end: List of indices of end of each desaturation event.
        CT_Threshold: Percentage of the time spent below the “CT_Threshold” % oxygen saturation level.
        CA_Baseline: Baseline to compute the CA feature. Default value is mean of the signal.

    :return:
        HypoxicBurdenMeasuresResults class containing the following features:
            -	CA: Integral SpO2 below the xx SpO2 level normalized by the total recording time
            -   CT: Percentage of the time spent below the xx% oxygen saturation level
            -   POD: Percentage of oxygen desaturation events
            -   AODmax: The area under the oxygen desaturation event curve, using the maximum SpO2 value as baseline
                and normalized by the total recording time
            -   AOD100: Cumulative area of desaturations under the 100% SpO2 level as baseline and normalized
                by the total recording time

    """

    check_shape(signal)

    desaturations = {'begin': begin, 'end': end}

    if CA_Baseline is None:
        CA_Baseline = np.nanmean(signal)

    return CompHBMeasures_(signal, desaturations, CT_Threshold, CA_Baseline)


def CA(signal, CA_baseline=None) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        CA_Baseline: Baseline to compute the CA feature. Default value is mean of the signal.

    :return:
        CA: Integral SpO2 below the xx SpO2 level normalized by the total recording time

    """

    check_shape(signal)

    if CA_baseline is None:
        CA_baseline = np.mean(signal)
    return CompCA_(signal, CA_baseline)


def CT(signal, CT_baseline=90) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        CA_Baseline: Baseline to compute the CA feature. Default value is mean of the signal.

    :return:
        CT: Percentage of the time spent below the xx% oxygen saturation level

    """

    check_shape(signal)

    return CompCT_(signal, CT_baseline)
