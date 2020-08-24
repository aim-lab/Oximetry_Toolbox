import numpy as np
from _spo2._Detector import _sc_resamp_, _sc_median_


def SetRange(signal, Range_min=50, Range_max=100):
    """
        Range function. Remove values lower than 50 or greater than 100, considered as non-physiological

        :param
            signal: 1-d array, of shape (N,) where N is the length of the signal

        :return:
            preprocessed signal, 1-d numpy array.
    """

    # return np.delete(signal, np.argwhere((signal >= Range_max) | (signal <= Range_min)))
    signal = np.array(signal)
    signal = np.ma.masked_where(((signal < Range_min) | (signal > Range_max)), signal).tolist()
    signal = [x if x is not None else np.nan for x in signal]
    signal = np.array(signal)
    return signal


def ResampSpO2(signal, OriginalFreq):
    """
        Resample the SpO2 signal to 1Hz.

        :param
            signal: 1-d array, of shape (N,) where N is the length of the signal
            OriginalFreq: the original frequency.

        :return:
            resampled signal, 1-d numpy array.
    """

    return _sc_resamp_(signal, OriginalFreq)


def DeltaFilter(signal, Diff=4):
    """
        Apply Delta Filter to the signal.

        :param
            signal: 1-d array, of shape (N,) where N is the length of the signal
            Diff: parameter of the delta filter.

        :return:
            preprocessed signal, 1-d numpy array.
    """

    signal_filtered = []
    for i, data in enumerate(signal):
        if i == 0:
            signal_filtered.append(data)
        else:
            if (((signal_filtered[-1] - data) / signal_filtered[-1]) * 100) < Diff:
                signal_filtered.append(data)
    return signal_filtered


def MedianSpO2(signal, FilterLength=9):
    """
        Apply a median filter to the SpO2 signal.

        :param
            signal: 1-d array, of shape (N,) where N is the length of the signal
            FilterLength (Optional): The length of the filter.

        :return:
            preprocessed signal, 1-d numpy array.
    """

    return _sc_median_(signal, medfilt_lg=FilterLength)


def BlockData(signal, treshold=50):
    """
        Apply a block data filter to the SpO2 signal.

        :param
            signal: 1-d array, of shape (N,) where N is the length of the signal
            treshold (Optional): treshold parameter for block data filter.

        :return:
            preprocessed signal, 1-d numpy array.
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
