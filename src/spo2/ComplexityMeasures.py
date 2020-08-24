from _spo2._ComplexityMeasures import _CompApEn_, _CompLZ_, _CompSampEn_, _CompDFA_, _CompCTM_
from _spo2._ErrorHandler import _check_shape_
from _spo2._ResultsClasses import ComplexityMeasuresResults


def ComplexityMeasures(signal, CTM_Threshold=0.25, DFA_Window=20, M_Sampen=3, R_Sampen=0.2) -> ComplexityMeasuresResults:
    """
    Function that calculates Complexity Features from spo2 time series.
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        CTM_Threshold: Radius of Central Tendency Measure.
        DFA_Window: Length of window to calculate DFA biomarker.
        M_Sampen: Embedding dimension to compute SampEn.
        R_Sampen: Tolerance to compute SampEn.

    :return:
        ComplexityMeasuresResults class containing the following features:
            -	ApEn: Approximate Entropy.
            -   LZ: Lempel-Ziv complexity.
            -	CTM: Central Tendency Measure.
            -   SampEn: Sample Entropy.
            -	DFA: Detrended Fluctuation Analysis.
    """

    _check_shape_(signal)

    return ComplexityMeasuresResults(_CompApEn_(signal[100:1000]), _CompLZ_(signal), _CompCTM_(signal, CTM_Threshold),
                                     _CompSampEn_(signal[100:1000], M_Sampen, R_Sampen), _CompDFA_(signal, DFA_Window))


def ApEn(signal) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Approximate Entropy.
    """

    _check_shape_(signal)

    return _CompApEn_(signal)


def LZ(signal) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Lempel-Ziv complexity.
    """

    _check_shape_(signal)

    return _CompLZ_(signal)


def CTM(signal, CTM_Threshold=0.25) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Central Tendency Measure.
    """

    _check_shape_(signal)

    return _CompCTM_(signal, CTM_Threshold)


def SampEn(signal, M_sampen=3, R_sampen=0.2) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal
        M_sampen: embedding dimension to compute SampEn.
        R_sampen: tolerance to compute SampEn.

    :return:
        Sample Entropy.
    """

    _check_shape_(signal)

    return _CompSampEn_(signal, M_sampen, R_sampen)


def DFA(signal, DFA_Window=20) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Detrended Fluctuation Analysis.
    """

    _check_shape_(signal)

    return _CompDFA_(signal, DFA_Window)
