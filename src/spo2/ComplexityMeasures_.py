import numpy as np
from lempel_ziv_complexity import lempel_ziv_complexity
from scipy import integrate, stats

from ErrorHandler import check_len_ApEn


def CompApEn_(signal):
    # Reference: Utility of Approximate Entropy From Overnight Pulse Oximetry Data
    #               in the Diagnosis of the Obstructive Sleep Apnea Syndrome
    signal = np.array(signal)
    signal = signal[np.logical_not(np.isnan(signal))]

    phi_m = ApEN_m_(2, signal)
    phi_m1 = ApEN_m_(3, signal)
    with np.errstate(invalid='ignore'):
        res = phi_m - phi_m1

    return res


def ApEN_m_(m, signal):
    # Help function of CompApEn
    N = len(signal)
    r = 0.25 * np.nanstd(signal)
    check_len_ApEn(N, m)
    C = np.zeros(shape=(N - m + 1))

    res = [signal[i:i + m] for i in range(0, N - m + 1)]
    for i in range(0, N - m + 1):
        C[i] = np.nansum(dist_(res[i], res, r)) / (N - m + 1)

    phi_m = np.nansum(np.log(C)) / (N - m + 1)
    return phi_m


def dist_(window1, window2, r):
    # Help function of ApEN_m
    window1 = np.array(window1)
    window2 = np.array(window2)

    with np.errstate(invalid='ignore'):
        return np.nanmax(abs(window1 - window2), axis=1) < r


def CompLZ_(signal):
    # Reference :Non-linear characteristics of blood oxygen saturation from nocturnal oximetry
    #           for obstructive sleep apnoea detection
    median = np.median(signal)
    res = [signal[i] > median for i in range(0, len(signal))]
    byte = [str(int(b is True)) for b in res]
    return lempel_ziv_complexity(''.join(byte))


def CompCTM_(signal, p):
    # Reference: Non-linear characteristics of blood oxygen saturation from nocturnal oximetry
    #           for obstructive sleep apnoea detection
    res = 0
    for i in range(len(signal) - 2):
        res += d_CTM_(i, p, signal)
    return res / (len(signal) - 2)


def d_CTM_(i, p, signal):
    # Help function of CompCTM
    if np.sqrt(((signal[i + 2] - signal[i + 1]) ** 2) + ((signal[i + 1] - signal[i]) ** 2)) < p:
        return 1
    return 0


def CompSampEn_(signal, m=3, r=0.2):
    N = len(signal)

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


def CompDFA_(signal, n=20):
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
        slope, intercept, _, _, _ = stats.linregress(x, y[i:i + n])
        least_square[i:i + n] = slope * x + intercept
        i += n

    return np.sqrt(np.nansum((y - least_square) ** 2) / len(y))
