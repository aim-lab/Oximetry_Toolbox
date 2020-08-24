from ComplexityMeasures_ import CompApEn_, CompLZ_, CompSampEn_, CompDFA_, CompCTM_
from ErrorHandler import check_shape
from ResultsClasses import ComplexityMeasuresResults


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

    check_shape(signal)

    return ComplexityMeasuresResults(CompApEn_(signal[100:1000]), CompLZ_(signal), CompCTM_(signal, CTM_Threshold),
                                     CompSampEn_(signal[100:1000], M_Sampen, R_Sampen), CompDFA_(signal, DFA_Window))


def ApEn(signal) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Approximate Entropy.
    """

    check_shape(signal)

    return CompApEn_(signal)


def LZ(signal) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Lempel-Ziv complexity.
    """

    check_shape(signal)

    return CompLZ_(signal)


def CTM(signal, CTM_Threshold=0.25) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Central Tendency Measure.
    """

    check_shape(signal)

    return CompCTM_(signal, CTM_Threshold)


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

    check_shape(signal)

    return CompSampEn_(signal, M_sampen, R_sampen)


def DFA(signal, DFA_Window=20) -> float:
    """
    Suppose that the data has been preprocessed.

    :param
        signal: 1-d array, of shape (N,) where N is the length of the signal

    :return:
        Detrended Fluctuation Analysis.
    """

    check_shape(signal)

    return CompDFA_(signal, DFA_Window)
