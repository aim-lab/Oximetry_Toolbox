from _spo2._DesaturationsMeasures import _processing_desat_
from _spo2._ErrorHandler import _check_shape_
from _spo2._ResultsClasses import DesaturationsMeasuresResults


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

    _check_shape_(signal)

    desaturations = {'begin': begin, 'end': end}

    return _processing_desat_(signal, desaturations)
