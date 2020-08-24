from _spo2._ErrorHandler import _check_shape_
from _spo2._PeriodicityMeasures import _PRSAFeatures_, _SpectralAnalysis_
from _spo2._ResultsClasses import PRSAResults, PSDResults


def PRSAMeasures(signal, PRSA_Window=10, K_AC=2) -> PRSAResults:
    """
    Function that calculates PRSA Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        PRSA_Window: Fragment duration of PRSA.
        K_AC: Number of values to shift when computing autocorrelation

    :return:
        PRSAResults class containing the following features:
            -	PRSAc: PRSA capacity.
            -	PRSAad: PRSA amplitude difference.
            -	PRSAos: PRSA overall slope.
            -	PRSAsb: PRSA slope before the anchor point.
            -	PRSAsa: PRSA slope after the anchor point.
            -	AC: Autocorrelation.
    """

    _check_shape_(signal)

    return _PRSAFeatures_(signal, PRSA_Window, K_AC)


def PSDMeasures(signal) -> PSDResults:
    """
    Function that calculates PSD Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: The SpO2 signal, of shape (N,)

    :return:
        PSDResults class containing the following features:
            -   PSD_total: The amplitude of the spectral signal.
            -   PSD_band: The amplitude of the signal multiplied by a band-pass filter between 0.014 and 0.033 Hz.
            -   PSD_ratio: The ratio between PSD_total and PSD_band.
            -   PDS_peak: The max value of the FFT into the band 0.014-0.033 Hz.
    """

    _check_shape_(signal)

    return _SpectralAnalysis_(signal)
