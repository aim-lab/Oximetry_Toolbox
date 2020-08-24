import numpy as np

from ErrorHandler import check_shape
from OverallGeneralMeasures_ import ApplyPercentile, BelowMedian_, NumZC_, DeltaIndex_, ComputeRange
from ResultsClasses import OverallGeneralMeasuresResult


def OverallGeneralMeasures(signal, ZC_Baseline=None, percentile=1, M_Threshold=2, DI_Window=12) \
        -> OverallGeneralMeasuresResult:
    """
    Function that calculates Overall General Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        ZC_Baseline: Baseline for calculating number of zero-crossing points.
        percentile: Percentile to perform. For example, for percentile 1, the argument should be 1
        M_Threshold: Percentage of the signal M_Threshold % below median oxygen saturation. Typically use 1,2 or 5

    :return:
        OveralGeneralMeasuresResult class containing the following features:
            -	AV: Average of the signal.
            -	MED: Median of the signal.
            -	Min: Minimum value of the signal.
            -	SD: Std of the signal.
            -	RG: SpO2 range (difference between the max and min value).
            -	P: percentile.
            -	M: Percentage of the signal x% below median oxygen saturation.
            -	ZC: Number of zero-crossing points.
            -	DI: Delta Index.
    """
    check_shape(signal)

    if ZC_Baseline is None:
        ZC_Baseline = np.nanmean(signal)

    return OverallGeneralMeasuresResult(np.nanmean(signal), np.nanmedian(signal), np.nanmin(signal), np.nanstd(signal),
                                        ComputeRange(signal), ApplyPercentile(signal, percentile),
                                        BelowMedian_(signal, M_Threshold), NumZC_(signal, ZC_Baseline),
                                        DeltaIndex_(signal, DI_Window))


def BelowMedian(signal, M_Threshold=2) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        M_Threshold: Percentage of the signal M_Threshold % below median oxygen saturation. Typically use 1,2 or 5

    :return:
        M: Percentage of the signal x% below median oxygen saturation.
    """

    check_shape(signal)

    return BelowMedian_(signal, M_Threshold)


def NumZC(signal, ZC_Baseline=None) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        ZC_Baseline: Baseline for calculating number of zero-crossing points.

    :return:
        ZC: Number of zero-crossing points.
    """

    check_shape(signal)

    if ZC_Baseline is None:
        ZC_Baseline = np.nanmean(signal)
    return NumZC_(signal, ZC_Baseline)


def DeltaIndex(signal, DI_Window=12) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        DI_Window: Window to calculate DelTa Index.

    :return:
        DI: Delta Index.
    """

    check_shape(signal)

    return DeltaIndex_(signal, DI_Window)
