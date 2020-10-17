import numpy as np
from scipy import integrate, stats
import warnings
from lempel_ziv_complexity import lempel_ziv_complexity

from pobm._ErrorHandler import _check_len_ApEn_
from pobm.obm.desat import DesaturationsMeasures


def odi(signal, ODI_Threshold=3):
    """
    Calculates the ODI from spo2 time series.
    Suppose that the data has been preprocessed.

    :param signal: The SpO2 signal, of shape (N,)
    :param ODI_Threshold: Threshold to compute Oxygen Desaturation Index.
    :return: ODI
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    desat_class = DesaturationsMeasures(ODI_Threshold)
    return desat_class.desaturation_detector(signal)


def sampen(signal, M_Sampen=3, R_Sampen=0.2):
    """
    Compute the Sample Entropy
    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :param M_Sampen: Embedding dimension to compute SampEn.
    :param R_Sampen: Tolerance to compute SampEn.
    :return: SampEn
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    N = len(signal)
    m = M_Sampen
    r = R_Sampen

    # Split time series and save all templates of length m
    xmi = np.array([signal[i: i + m] for i in range(N - m)])
    xmj = np.array([signal[i: i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    with np.errstate(invalid='ignore'):
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([signal[i: i + m] for i in range(N - m + 1)])

    with np.errstate(invalid='ignore'):
        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)


def lempel_ziv(signal):
    """
    Compute lempel-ziv, according to the paper
    Non-linear characteristics of blood oxygen saturation from nocturnal oximetry
    for obstructive sleep apnoea detection
    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :return: LZ
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    median = np.median(signal)
    res = [signal[i] > median for i in range(0, len(signal))]
    byte = [str(int(b is True)) for b in res]
    return lempel_ziv_complexity(''.join(byte))


def apen(signal, M_ApEn=2, R_ApEn=0.25):
    """
    Compute the approximate entropy, according to the paper
    Utility of Approximate Entropy From Overnight Pulse Oximetry Data in the Diagnosis
    of the Obstructive Sleep Apnea Syndrome
    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :return: ApEn
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    signal = np.array(signal)
    signal = signal[np.logical_not(np.isnan(signal))]

    phi_m = __apen(M_ApEn, R_ApEn, signal)
    phi_m1 = __apen(M_ApEn + 1, R_ApEn, signal)
    with np.errstate(invalid='ignore'):
        res = phi_m - phi_m1

    return res


def __apen(m, R_ApEn, signal):
    """
    Help function of CompApEn
    """
    N = len(signal)
    r = R_ApEn
    _check_len_ApEn_(N, m)
    C = np.zeros(shape=(N - m + 1))

    res = [signal[i:i + m] for i in range(0, N - m + 1)]
    for i in range(0, N - m + 1):
        C[i] = np.nansum(__dist(res[i], res, r)) / (N - m + 1)

    phi_m = np.nansum(np.log(C)) / (N - m + 1)
    return phi_m


def __dist(window1, window2, r):
    """
    Help function of CompApEn
    """
    window1 = np.array(window1)
    window2 = np.array(window2)

    with np.errstate(invalid='ignore'):
        return np.nanmax(abs(window1 - window2), axis=1) < r


def dfa(signal, DFA_Window=20):
    """
    Compute DFA
    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :param DFA_Window: Length of window to calculate DFA biomarker.
    :return: DFA
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    n = DFA_Window
    y = integrate.cumtrapz(signal - np.nanmean(signal))
    least_square = np.zeros(len(y))

    i = 0
    while i < len(y):
        if i + n > len(y):
            n = len(y) - i
        if n == 1:
            least_square[-1] = y[-1]
            break
        x = np.array(range(0, n))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            slope, intercept, _, _, _ = stats.linregress(x, y[i:i + n])
        least_square[i:i + n] = slope * x + intercept
        i += n

    return np.sqrt(np.nansum((y - least_square) ** 2) / len(y))
