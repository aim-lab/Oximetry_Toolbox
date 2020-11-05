import numpy as np
from scipy import signal


def set_range(signal, Range_min=50, Range_max=100):
    """
    Range function. Remove values lower than Range_min or greater than Range_max, considered as non-physiological

    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :param Range_min: minimum value for removing the data
    :type Range_min: int, optional
    :param Range_max: maximum value for removing the data
    :type Range_max: int, optional

    :return: preprocessed signal, 1-d numpy array.

    """

    # return np.delete(signal, np.argwhere((signal >= Range_max) | (signal <= Range_min)))
    signal = np.array(signal)
    signal = np.ma.masked_where(((signal < Range_min) | (signal > Range_max)), signal).tolist()
    signal = [x if x is not None else np.nan for x in signal]
    signal = np.array(signal)
    return signal


def resamp_spo2(signal, OriginalFreq):
    """
    Resample the SpO2 signal to 1Hz.
    Assumption: any missing/abnormal values are represented as 'np.nan'

    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :param OriginalFreq: the original frequency.

    :return: resampled signal, 1-d numpy array, the resampled spo2 time series at 1Hz

    """

    len_in = len(signal)
    len_out = round(len_in / OriginalFreq)
    data_out = []
    for jj in range(len_out):
        data_out = np.append(data_out, np.median(signal[jj * OriginalFreq:(jj + 1) * OriginalFreq]))

    return data_out


def dfilter(signal, Diff=4):
    """
    Apply Delta Filter to the signal.

    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :param Diff: parameter of the delta filter.
    :type Diff: int, optional

    :return: preprocessed signal, 1-d numpy array.

    """

    signal_filtered = []
    for i, data in enumerate(signal):
        if i == 0:
            signal_filtered.append(data)
        else:
            if (((signal_filtered[-1] - data) / signal_filtered[-1]) * 100) < Diff:
                signal_filtered.append(data)
    return signal_filtered


def median_spo2(signal_spo2, FilterLength=9):
    """
    Apply a median filter to the SpO2 signal.
    Median filter used to smooth the spo2 time series and avoid sporadic increase/decrease of spo2 which could 
    affect the detection of the desaturations.
    Assumption: any missing/abnormal values are represented as 'np.nan'

    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :param FilterLength (Optional): The length of the filter.

    :return: preprocessed signal, 1-d numpy array.

    """

    data_med = signal.medfilt(np.round(signal_spo2), FilterLength)

    return data_med


def block_data(signal, treshold=50):
    """
    Apply a block data filter to the SpO2 signal.

    :param signal: 1-d array, of shape (N,) where N is the length of the signal
    :param treshold: treshold parameter for block data filter.
    :type treshold: int, optional

    :return: preprocessed signal, 1-d numpy array.

    """
    signal = np.array(signal)
    mask = np.ones(len(signal), dtype=bool)
    for i, data in enumerate(signal):
        if data < treshold:
            mask[i - 10:i + 10] = False

    for i, data in enumerate(signal):
        if not mask[i]:
            signal[i] = None

    mean_signal = np.mean(signal)
    mask = np.ones(len(signal), dtype=bool)

    i = 0
    while i + 100 < len(signal):
        mean_block = np.mean(signal[i:i + 100])
        if mean_block < mean_signal * 0.94:
            mask[i:i + 100] = False
        i += 100

    for i, data in enumerate(signal):
        if not mask[i]:
            signal[i] = None

    return signal
