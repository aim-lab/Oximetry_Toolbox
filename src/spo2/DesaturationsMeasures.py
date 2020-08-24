from DesaturationsMeasures_ import processing_desat_
from ErrorHandler import check_shape
from ResultsClasses import DesaturationsMeasuresResults


def DesaturationsMeasures(signal, begin, end) -> DesaturationsMeasuresResults:
    """
    Function that calculates the Desaturation Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        begin: List of indices of beginning of each desaturation event.
        end: List of indices of end of each desaturation event.

    :return:
        DesaturationsMeasuresResults class containing the following features:
            -	DL_u: Mean of desaturation length
            -	DL_sd: Standard deviation of desaturation length
            -	DA100_u: Mean of desaturation area using 100% as baseline.
            -	DA100_sd: Standard deviation of desaturation area using 100% as baseline
            -	DAmax_u: Mean of desaturation area using max value as baseline.
            -	DAmax_sd: Standard deviation of desaturation area using max value as baseline
            -	DD100_u: Mean of depth desaturation from 100%.
            -	DD100_sd: Standard deviation of depth desaturation from 100%.
            -	DDmax_u: Mean of depth desaturation from max value.
            -	DDmax_sd: Standard deviation of depth desaturation from max value.
            -	DS_u: Mean of the desaturation slope.
            -	DS_sd: Standard deviation of the desaturation slope.
            -   TD_u: Mean of time between two consecutive desaturation events.
            -   TD_sd: Standard deviation of time between 2 consecutive desaturation events.
    """

    check_shape(signal)

    desaturations = {'begin': begin, 'end': end}

    return processing_desat_(signal, desaturations)
