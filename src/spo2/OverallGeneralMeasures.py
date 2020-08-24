import numpy as np

from _spo2._ErrorHandler import _check_shape_
from _spo2._OverallGeneralMeasures import _ApplyPercentile_, _BelowMedian_, _NumZC_, _DeltaIndex_, _ComputeRange_
from _spo2._ResultsClasses import OverallGeneralMeasuresResult


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
    _check_shape_(signal)

    if ZC_Baseline is None:
        ZC_Baseline = np.nanmean(signal)

    return OverallGeneralMeasuresResult(np.nanmean(signal), np.nanmedian(signal), np.nanmin(signal), np.nanstd(signal),
                                        _ComputeRange_(signal), _ApplyPercentile_(signal, percentile),
                                        _BelowMedian_(signal, M_Threshold), _NumZC_(signal, ZC_Baseline),
                                        _DeltaIndex_(signal, DI_Window))


def BelowMedian(signal, M_Threshold=2) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        M_Threshold: Percentage of the signal M_Threshold % below median oxygen saturation. Typically use 1,2 or 5

    :return:
        M: Percentage of the signal x% below median oxygen saturation.
    """

    _check_shape_(signal)

    return _BelowMedian_(signal, M_Threshold)


def NumZC(signal, ZC_Baseline=None) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        ZC_Baseline: Baseline for calculating number of zero-crossing points.

    :return:
        ZC: Number of zero-crossing points.
    """

    _check_shape_(signal)

    if ZC_Baseline is None:
        ZC_Baseline = np.nanmean(signal)
    return _NumZC_(signal, ZC_Baseline)


def DeltaIndex(signal, DI_Window=12) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        DI_Window: Window to calculate DelTa Index.

    :return:
        DI: Delta Index.
    """

    _check_shape_(signal)

    return _DeltaIndex_(signal, DI_Window)
